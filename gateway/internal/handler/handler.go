package handler

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"google.golang.org/grpc/codes"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/status"

	pb "github.com/Kobe16/crucible/gateway/gen/inference"
	"github.com/Kobe16/crucible/gateway/internal/batcher"
)

// requestIDKey is a context key for the per-request ID set by LoggingMiddleware.
type requestIDKey struct{}

// responseWriter wraps http.ResponseWriter to capture the status code written by handlers.
type responseWriter struct {
	http.ResponseWriter
	status int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.status = code
	rw.ResponseWriter.WriteHeader(code)
}

// LoggingMiddleware generates a request_id for every inbound request, stores it in
// the context, and logs a single structured http_request entry after the handler returns.
func LoggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestID := newRequestID()
		ctx := context.WithValue(r.Context(), requestIDKey{}, requestID)
		r = r.WithContext(ctx)
		rw := &responseWriter{ResponseWriter: w, status: http.StatusOK}
		start := time.Now()
		next.ServeHTTP(rw, r)
		slog.InfoContext(ctx, "http_request",
			"method", r.Method,
			"path", r.URL.Path,
			"status", rw.status,
			"latency_ms", time.Since(start).Milliseconds(),
			"request_id", requestID,
		)
	})
}

// StatusProbe is the worker-status dependency for the /health and /status
// endpoints. *inference.Client satisfies this interface.
type StatusProbe interface {
	CheckHealth(ctx context.Context) (*healthpb.HealthCheckResponse, error)
	GetWorkerStatus(ctx context.Context) (*pb.WorkerStatusResponse, error)
}

// Predictor routes a prediction request and returns its result.
// *batcher.Batcher satisfies this interface.
type Predictor interface {
	Submit(req *batcher.PendingRequest) batcher.Result
}

// Handler is an HTTP handler for inference requests and health checks.
type Handler struct {
	probe     StatusProbe
	predictor Predictor
}

func New(probe StatusProbe, predictor Predictor) *Handler {
	return &Handler{probe: probe, predictor: predictor}
}

// predictRequest is the expected JSON body for a prediction request.
type predictRequest struct {
	Input      string            `json:"input"`
	Parameters map[string]string `json:"parameters,omitempty"`
}

// predictResponse is the JSON response for a successful prediction request.
type predictResponse struct {
	RequestID string `json:"request_id"`
	Output    string `json:"output"`
}

// healthResponse is the JSON response for a health check request.
type healthResponse struct {
	Status string `json:"status"`
}

// statusResponse is the JSON response for the detailed worker status endpoint.
type statusResponse struct {
	Status          string  `json:"status"`
	InFlightBatches int32   `json:"in_flight_batches"`
	GpuUtilization  float32 `json:"gpu_utilization"`
}

// errorResponse is the JSON response for an error.
type errorResponse struct {
	Error string `json:"error"`
}

// Predict handles POST /predict requests, forwarding them to the batcher and returning the response.
func (h *Handler) Predict(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20) // 1 MB limit

	// Decode the JSON body into a predictRequest struct, handling invalid JSON and missing input.
	var req predictRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, errorResponse{Error: "invalid JSON body"})
		return
	}

	if req.Input == "" {
		writeJSON(w, http.StatusBadRequest, errorResponse{Error: "input is required"})
		return
	}

	// Retrieve the request ID set by LoggingMiddleware (falls back to empty string if middleware not used).
	requestID, _ := r.Context().Value(requestIDKey{}).(string)
	if requestID == "" {
		requestID = newRequestID()
	}

	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	// Build request and send it to batcher
	pending := &batcher.PendingRequest{
		RequestID:    requestID,
		Input:        req.Input,
		Parameters:   req.Parameters,
		ResponseChan: make(chan batcher.Result, 1),
		ArrivalTime:  time.Now(),
		Ctx:          ctx,
	}

	result := h.predictor.Submit(pending)

	// Request succeeded
	if result.Err == nil {
		writeJSON(w, http.StatusOK, predictResponse{
			RequestID: requestID,
			Output:    result.Output,
		})
		return
	}

	// Context error occurred (10s timeout fired)
	if errors.Is(result.Err, batcher.ErrQueueFull) {
		slog.WarnContext(r.Context(), "queue_full", "request_id", requestID)
		writeJSON(w, http.StatusServiceUnavailable, errorResponse{Error: "server too busy"})
		return
	}

	if errors.Is(result.Err, context.DeadlineExceeded) {
		slog.ErrorContext(r.Context(), "request_timeout", "error", result.Err, "request_id", requestID)
		writeJSON(w, http.StatusGatewayTimeout, errorResponse{Error: "request timed out"})
		return
	}

	// Request got cancelled
	if errors.Is(result.Err, context.Canceled) {
		slog.InfoContext(r.Context(), "request_cancelled", "request_id", requestID)
		writeJSON(w, 499, errorResponse{Error: "request cancelled"})
		return
	}

	// Everything else (gRPC error, missing-response error, etc.)
	code, httpStatus := mapGRPCError(result.Err)
	slog.ErrorContext(r.Context(), "predict_error", "grpc_code", code.String(), "error", result.Err, "request_id", requestID)
	writeJSON(w, httpStatus, errorResponse{Error: code.String()})
}

// Health handles GET /health requests using the standard grpc.health.v1 protocol.
func (h *Handler) Health(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	resp, err := h.probe.CheckHealth(ctx)
	if err != nil {
		writeJSON(w, http.StatusServiceUnavailable, healthResponse{Status: "unavailable"})
		return
	}

	if resp.Status == healthpb.HealthCheckResponse_SERVING {
		writeJSON(w, http.StatusOK, healthResponse{Status: "ok"})
	} else {
		writeJSON(w, http.StatusServiceUnavailable, healthResponse{Status: "unavailable"})
	}
}

// Status handles GET /status requests, returning detailed worker status including
// the application-level ServingStatus and metrics (in_flight_batches, gpu_utilization).
func (h *Handler) Status(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	resp, err := h.probe.GetWorkerStatus(ctx)
	if err != nil {
		writeJSON(w, http.StatusServiceUnavailable, errorResponse{Error: "worker unreachable"})
		return
	}

	writeJSON(w, http.StatusOK, statusResponse{
		Status:          resp.Status.String(),
		InFlightBatches: resp.InFlightBatches,
		GpuUtilization:  resp.GpuUtilization,
	})
}

// mapGRPCError converts a gRPC error to an HTTP status code and a gRPC code for logging.
func mapGRPCError(err error) (codes.Code, int) {
	st, ok := status.FromError(err)
	if !ok {
		return codes.Unknown, http.StatusInternalServerError
	}
	switch st.Code() {
	case codes.Unavailable:
		return codes.Unavailable, http.StatusServiceUnavailable
	case codes.InvalidArgument:
		return codes.InvalidArgument, http.StatusBadRequest
	case codes.Internal:
		return codes.Internal, http.StatusBadGateway
	case codes.DeadlineExceeded:
		return codes.DeadlineExceeded, http.StatusGatewayTimeout
	default:
		return st.Code(), http.StatusInternalServerError
	}
}

// newRequestID generates a random UUIDv4 string to use as a unique request ID for tracing.
func newRequestID() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	b[6] = (b[6] & 0x0f) | 0x40 // version 4
	b[8] = (b[8] & 0x3f) | 0x80 // variant 10
	return fmt.Sprintf("%08x-%04x-%04x-%04x-%012x",
		b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}

// writeJSON is a helper function to write a JSON response with the given status code and value.
func writeJSON(w http.ResponseWriter, statusCode int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		slog.Error("json_encode_error", "error", err)
	}
}
