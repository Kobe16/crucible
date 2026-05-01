package handler

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"google.golang.org/grpc/codes"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/status"

	pb "github.com/Kobe16/crucible/gateway/gen/inference"
	"github.com/Kobe16/crucible/gateway/internal/batcher"
)

// mockStatusProbe implements StatusProbe for testing.
type mockStatusProbe struct {
	checkHealthFn     func(ctx context.Context) (*healthpb.HealthCheckResponse, error)
	getWorkerStatusFn func(ctx context.Context) (*pb.WorkerStatusResponse, error)
}

func (m *mockStatusProbe) CheckHealth(ctx context.Context) (*healthpb.HealthCheckResponse, error) {
	if m.checkHealthFn != nil {
		return m.checkHealthFn(ctx)
	}
	return nil, nil
}

func (m *mockStatusProbe) GetWorkerStatus(ctx context.Context) (*pb.WorkerStatusResponse, error) {
	if m.getWorkerStatusFn != nil {
		return m.getWorkerStatusFn(ctx)
	}
	return nil, nil
}

// mockPredictor implements Predictor for testing.
type mockPredictor struct {
	submitFn func(req *batcher.PendingRequest) batcher.Result
}

func (m *mockPredictor) Submit(req *batcher.PendingRequest) batcher.Result {
	if m.submitFn != nil {
		return m.submitFn(req)
	}
	return batcher.Result{}
}

// TestPredict tests the Predict handler (/predict endpoint) by driving the
// prediction path through a mock Predictor. Validation errors short-circuit
// before Submit, so those test cases leave submitFn nil.
func TestPredict(t *testing.T) {
	tests := []struct {
		name       string
		body       string
		submitFn   func(req *batcher.PendingRequest) batcher.Result
		wantStatus int
		wantBody   string // substring match
	}{
		{
			name: "valid request",
			body: `{"input":"hello world"}`,
			submitFn: func(_ *batcher.PendingRequest) batcher.Result {
				return batcher.Result{Output: "POSITIVE (0.9500)"}
			},
			wantStatus: http.StatusOK,
			wantBody:   `"output":"POSITIVE (0.9500)"`,
		},
		{
			name: "valid request with parameters",
			body: `{"input":"hello","parameters":{"temp":"0.7"}}`,
			submitFn: func(req *batcher.PendingRequest) batcher.Result {
				if req.Parameters["temp"] != "0.7" {
					t.Errorf("expected temp=0.7, got %s", req.Parameters["temp"])
				}
				return batcher.Result{Output: "ok"}
			},
			wantStatus: http.StatusOK,
			wantBody:   `"output":"ok"`,
		},
		{
			name:       "empty input",
			body:       `{"input":""}`,
			wantStatus: http.StatusBadRequest,
			wantBody:   `"input is required"`,
		},
		{
			name:       "missing input field",
			body:       `{}`,
			wantStatus: http.StatusBadRequest,
			wantBody:   `"input is required"`,
		},
		{
			name:       "invalid JSON",
			body:       `{bad json`,
			wantStatus: http.StatusBadRequest,
			wantBody:   `"invalid JSON body"`,
		},
		{
			name: "worker unavailable",
			body: `{"input":"test"}`,
			submitFn: func(_ *batcher.PendingRequest) batcher.Result {
				return batcher.Result{Err: status.Error(codes.Unavailable, "connection refused")}
			},
			wantStatus: http.StatusServiceUnavailable,
		},
		{
			name: "worker internal error",
			body: `{"input":"test"}`,
			submitFn: func(_ *batcher.PendingRequest) batcher.Result {
				return batcher.Result{Err: status.Error(codes.Internal, "inference failed")}
			},
			wantStatus: http.StatusBadGateway,
		},
		{
			name: "worker deadline exceeded (gRPC)",
			body: `{"input":"test"}`,
			submitFn: func(_ *batcher.PendingRequest) batcher.Result {
				return batcher.Result{Err: status.Error(codes.DeadlineExceeded, "timeout")}
			},
			wantStatus: http.StatusGatewayTimeout,
		},
		{
			name: "worker invalid argument",
			body: `{"input":"test"}`,
			submitFn: func(_ *batcher.PendingRequest) batcher.Result {
				return batcher.Result{Err: status.Error(codes.InvalidArgument, "bad input")}
			},
			wantStatus: http.StatusBadRequest,
		},
		{
			name: "request context deadline exceeded",
			body: `{"input":"test"}`,
			submitFn: func(_ *batcher.PendingRequest) batcher.Result {
				return batcher.Result{Err: context.DeadlineExceeded}
			},
			wantStatus: http.StatusGatewayTimeout,
			wantBody:   `"request timed out"`,
		},
		{
			name: "missing response from worker",
			body: `{"input":"test"}`,
			submitFn: func(_ *batcher.PendingRequest) batcher.Result {
				return batcher.Result{Err: errors.New("worker did not return a response for request abc")}
			},
			wantStatus: http.StatusInternalServerError,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := New(&mockStatusProbe{}, &mockPredictor{submitFn: tt.submitFn})

			req := httptest.NewRequest(http.MethodPost, "/predict", strings.NewReader(tt.body))
			req.Header.Set("Content-Type", "application/json")
			rec := httptest.NewRecorder()

			h.Predict(rec, req)

			if rec.Code != tt.wantStatus {
				t.Errorf("status = %d, want %d", rec.Code, tt.wantStatus)
			}
			if tt.wantBody != "" && !strings.Contains(rec.Body.String(), tt.wantBody) {
				t.Errorf("body = %q, want substring %q", rec.Body.String(), tt.wantBody)
			}
		})
	}
}

// TestHealth tests the Health handler (/health endpoint) with different health check responses from the worker, including serving, not serving, and error scenarios.
func TestHealth(t *testing.T) {
	tests := []struct {
		name       string
		healthFn   func(ctx context.Context) (*healthpb.HealthCheckResponse, error)
		wantStatus int
		wantBody   string
	}{
		{
			name: "serving",
			healthFn: func(_ context.Context) (*healthpb.HealthCheckResponse, error) {
				return &healthpb.HealthCheckResponse{Status: healthpb.HealthCheckResponse_SERVING}, nil
			},
			wantStatus: http.StatusOK,
			wantBody:   `"status":"ok"`,
		},
		{
			name: "not serving",
			healthFn: func(_ context.Context) (*healthpb.HealthCheckResponse, error) {
				return &healthpb.HealthCheckResponse{Status: healthpb.HealthCheckResponse_NOT_SERVING}, nil
			},
			wantStatus: http.StatusServiceUnavailable,
			wantBody:   `"status":"unavailable"`,
		},
		{
			name: "error",
			healthFn: func(_ context.Context) (*healthpb.HealthCheckResponse, error) {
				return nil, errors.New("connection refused")
			},
			wantStatus: http.StatusServiceUnavailable,
			wantBody:   `"status":"unavailable"`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mock := &mockStatusProbe{checkHealthFn: tt.healthFn}
			h := New(mock, &mockPredictor{})

			req := httptest.NewRequest(http.MethodGet, "/health", nil)
			rec := httptest.NewRecorder()

			h.Health(rec, req)

			if rec.Code != tt.wantStatus {
				t.Errorf("status = %d, want %d", rec.Code, tt.wantStatus)
			}
			if !strings.Contains(rec.Body.String(), tt.wantBody) {
				t.Errorf("body = %q, want substring %q", rec.Body.String(), tt.wantBody)
			}
		})
	}
}

// TestStatus tests the Status handler (/status endpoint) with the two worker status responses: normal status and worker unreachable.
func TestStatus(t *testing.T) {
	tests := []struct {
		name       string
		statusFn   func(ctx context.Context) (*pb.WorkerStatusResponse, error)
		wantStatus int
		wantBody   string
	}{
		{
			name: "ok",
			statusFn: func(_ context.Context) (*pb.WorkerStatusResponse, error) {
				return &pb.WorkerStatusResponse{
					Status:          pb.ServingStatus_SERVING_STATUS_OK,
					InFlightBatches: 2,
					GpuUtilization:  0.75,
				}, nil
			},
			wantStatus: http.StatusOK,
			wantBody:   `"status":"SERVING_STATUS_OK"`,
		},
		{
			name: "worker unreachable",
			statusFn: func(_ context.Context) (*pb.WorkerStatusResponse, error) {
				return nil, errors.New("connection refused")
			},
			wantStatus: http.StatusServiceUnavailable,
			wantBody:   `"worker unreachable"`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mock := &mockStatusProbe{getWorkerStatusFn: tt.statusFn}
			h := New(mock, &mockPredictor{})

			req := httptest.NewRequest(http.MethodGet, "/status", nil)
			rec := httptest.NewRecorder()

			h.Status(rec, req)

			if rec.Code != tt.wantStatus {
				t.Errorf("status = %d, want %d", rec.Code, tt.wantStatus)
			}
			if !strings.Contains(rec.Body.String(), tt.wantBody) {
				t.Errorf("body = %q, want substring %q", rec.Body.String(), tt.wantBody)
			}
		})
	}
}

// TestMapGRPCError tests the mapGRPCError function with different gRPC/non-gRPC errors, verifying that the correct gRPC codes and corresponding HTTP status codes are returned for each case.
func TestMapGRPCError(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		wantCode codes.Code
		wantHTTP int
	}{
		{"unavailable", status.Error(codes.Unavailable, ""), codes.Unavailable, http.StatusServiceUnavailable},
		{"invalid argument", status.Error(codes.InvalidArgument, ""), codes.InvalidArgument, http.StatusBadRequest},
		{"internal", status.Error(codes.Internal, ""), codes.Internal, http.StatusBadGateway},
		{"deadline exceeded", status.Error(codes.DeadlineExceeded, ""), codes.DeadlineExceeded, http.StatusGatewayTimeout},
		{"other grpc code", status.Error(codes.NotFound, ""), codes.NotFound, http.StatusInternalServerError},
		{"non-grpc error", errors.New("plain error"), codes.Unknown, http.StatusInternalServerError},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			code, httpStatus := mapGRPCError(tt.err)
			if code != tt.wantCode {
				t.Errorf("code = %v, want %v", code, tt.wantCode)
			}
			if httpStatus != tt.wantHTTP {
				t.Errorf("httpStatus = %d, want %d", httpStatus, tt.wantHTTP)
			}
		})
	}
}

// TestLoggingMiddleware tests to make sure the LoggingMiddleware adds request ID to context and calls next handler.
func TestLoggingMiddleware(t *testing.T) {
	var gotRequestID string
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotRequestID, _ = r.Context().Value(requestIDKey{}).(string)
		w.WriteHeader(http.StatusCreated)
	})

	handler := LoggingMiddleware(inner)
	req := httptest.NewRequest(http.MethodGet, "/test", nil)
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if gotRequestID == "" {
		t.Error("expected non-empty request ID in context")
	}
	if rec.Code != http.StatusCreated {
		t.Errorf("status = %d, want %d", rec.Code, http.StatusCreated)
	}
}

// TestNewRequestID tests the newRequestID function to ensure it generates non-empty, unique IDs in the expected UUIDv4 format.
func TestNewRequestID(t *testing.T) {
	id1 := newRequestID()
	id2 := newRequestID()

	if id1 == "" {
		t.Error("expected non-empty request ID")
	}
	if id1 == id2 {
		t.Errorf("expected unique IDs, got %q twice", id1)
	}
	// UUIDv4 format: 8-4-4-4-12 hex chars
	if len(id1) != 36 {
		t.Errorf("expected 36-char UUID, got %d chars: %q", len(id1), id1)
	}
}

// TestStatusResponseJSON tests the JSON encoding of the status response from the /status endpoint.
func TestStatusResponseJSON(t *testing.T) {
	// Verify the status endpoint returns correct numeric fields
	mock := &mockStatusProbe{
		getWorkerStatusFn: func(_ context.Context) (*pb.WorkerStatusResponse, error) {
			return &pb.WorkerStatusResponse{
				Status:          pb.ServingStatus_SERVING_STATUS_OK,
				InFlightBatches: 5,
				GpuUtilization:  0.42,
			}, nil
		},
	}
	h := New(mock, &mockPredictor{})
	req := httptest.NewRequest(http.MethodGet, "/status", nil)
	rec := httptest.NewRecorder()
	h.Status(rec, req)

	var resp statusResponse
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.InFlightBatches != 5 {
		t.Errorf("in_flight_batches = %d, want 5", resp.InFlightBatches)
	}
	if resp.GpuUtilization < 0.41 || resp.GpuUtilization > 0.43 {
		t.Errorf("gpu_utilization = %f, want ~0.42", resp.GpuUtilization)
	}
}
