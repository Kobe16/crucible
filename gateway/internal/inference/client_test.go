package inference

import (
	"context"
	"net"
	"testing"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/health"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"

	pb "github.com/Kobe16/crucible/gateway/gen/inference"
)

const bufSize = 1024 * 1024

// fakeInferenceServer implements InferenceServiceServer for testing.
type fakeInferenceServer struct {
	pb.UnimplementedInferenceServiceServer
	batchFn      func(context.Context, *pb.BatchRequest) (*pb.BatchResponse, error)
	workerStatus *pb.WorkerStatusResponse
}

func (f *fakeInferenceServer) BatchInference(ctx context.Context, req *pb.BatchRequest) (*pb.BatchResponse, error) {
	if f.batchFn != nil {
		return f.batchFn(ctx, req)
	}
	return &pb.BatchResponse{}, nil
}

func (f *fakeInferenceServer) GetWorkerStatus(_ context.Context, _ *pb.WorkerStatusRequest) (*pb.WorkerStatusResponse, error) {
	if f.workerStatus != nil {
		return f.workerStatus, nil
	}
	return &pb.WorkerStatusResponse{Status: pb.ServingStatus_SERVING_STATUS_OK}, nil
}

// newTestClient creates a Client connected to an in-process gRPC server via bufconn.
// Buffer Connection is like a fake, in-memory network cable - it spins up a spin up a real gRPC server engine & gRPC client, but they talk through memory.
func newTestClient(t *testing.T, fake *fakeInferenceServer, healthServing bool) *Client {
	t.Helper()

	lis := bufconn.Listen(bufSize)
	srv := grpc.NewServer()
	pb.RegisterInferenceServiceServer(srv, fake)

	hsrv := health.NewServer()
	if healthServing {
		hsrv.SetServingStatus("", healthpb.HealthCheckResponse_SERVING)
	} else {
		hsrv.SetServingStatus("", healthpb.HealthCheckResponse_NOT_SERVING)
	}
	healthpb.RegisterHealthServer(srv, hsrv)

	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(func() { srv.Stop() })

	conn, err := grpc.NewClient("passthrough:///bufconn",
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) {
			return lis.DialContext(ctx)
		}),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		t.Fatalf("dial bufconn: %v", err)
	}
	t.Cleanup(func() { conn.Close() })

	return &Client{
		conn:       conn,
		stub:       pb.NewInferenceServiceClient(conn),
		healthStub: healthpb.NewHealthClient(conn),
	}
}

// TestInfer tests the Infer method of the Client with various scenarios, including successful inference, empty response, and server error.
func TestInfer(t *testing.T) {
	tests := []struct {
		name    string
		batchFn func(context.Context, *pb.BatchRequest) (*pb.BatchResponse, error)
		wantOut string
		wantErr bool
	}{
		{
			name: "success",
			batchFn: func(_ context.Context, req *pb.BatchRequest) (*pb.BatchResponse, error) {
				return &pb.BatchResponse{
					Responses: []*pb.InferenceResponse{
						{RequestId: req.Requests[0].RequestId, Output: "POSITIVE (0.95)"},
					},
				}, nil
			},
			wantOut: "POSITIVE (0.95)",
		},
		{
			name: "empty response",
			batchFn: func(_ context.Context, _ *pb.BatchRequest) (*pb.BatchResponse, error) {
				return &pb.BatchResponse{}, nil
			},
			wantErr: true,
		},
		{
			name: "server error",
			batchFn: func(_ context.Context, _ *pb.BatchRequest) (*pb.BatchResponse, error) {
				return nil, status.Error(codes.Unavailable, "worker down")
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fake := &fakeInferenceServer{batchFn: tt.batchFn}
			client := newTestClient(t, fake, true)

			out, err := client.Infer(context.Background(), "req-1", "hello", nil)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if out != tt.wantOut {
				t.Errorf("output = %q, want %q", out, tt.wantOut)
			}
		})
	}
}

// TestCheckHealth tests the CheckHealth method of the Client for both serving and not serving scenarios.
func TestCheckHealth(t *testing.T) {
	tests := []struct {
		name    string
		serving bool
		want    healthpb.HealthCheckResponse_ServingStatus
	}{
		{"serving", true, healthpb.HealthCheckResponse_SERVING},
		{"not serving", false, healthpb.HealthCheckResponse_NOT_SERVING},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fake := &fakeInferenceServer{}
			client := newTestClient(t, fake, tt.serving)

			resp, err := client.CheckHealth(context.Background())
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if resp.Status != tt.want {
				t.Errorf("status = %v, want %v", resp.Status, tt.want)
			}
		})
	}
}

// TestGetWorkerStatus tests the GetWorkerStatus method of the Client, verifying that it correctly gets worker status info from the server.
func TestGetWorkerStatus(t *testing.T) {
	fake := &fakeInferenceServer{
		workerStatus: &pb.WorkerStatusResponse{
			Status:          pb.ServingStatus_SERVING_STATUS_DEGRADED,
			InFlightBatches: 3,
			GpuUtilization:  0.85,
		},
	}
	client := newTestClient(t, fake, true)

	resp, err := client.GetWorkerStatus(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Status != pb.ServingStatus_SERVING_STATUS_DEGRADED {
		t.Errorf("status = %v, want DEGRADED", resp.Status)
	}
	if resp.InFlightBatches != 3 {
		t.Errorf("in_flight_batches = %d, want 3", resp.InFlightBatches)
	}
}
