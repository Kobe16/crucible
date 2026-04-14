package worker

import (
	"context"
	"fmt"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"

	pb "github.com/Kobe16/crucible/gateway/gen/inference"
)

// Client holds the gRPC client connection and stub for communicating with the worker.
type Client struct {
	conn       *grpc.ClientConn
	stub       pb.InferenceServiceClient
	healthStub healthpb.HealthClient
}

// NewClient creates a new gRPC client for the worker at the specified address.
func NewClient(addr string) (*Client, error) {
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("grpc new client: %w", err)
	}
	return &Client{
		conn:       conn,
		stub:       pb.NewInferenceServiceClient(conn),
		healthStub: healthpb.NewHealthClient(conn),
	}, nil
}

// Infer creates a batch inference request, sends it to the worker, and returns the output.
func (c *Client) Infer(ctx context.Context, requestID, input string, params map[string]string) (string, error) {
	req := &pb.BatchRequest{
		Requests: []*pb.InferenceRequest{
			{
				RequestId:  requestID,
				Input:      input,
				Parameters: params,
			},
		},
	}

	resp, err := c.stub.BatchInference(ctx, req)
	if err != nil {
		return "", err
	}

	if len(resp.Responses) == 0 {
		return "", fmt.Errorf("worker returned empty batch response")
	}

	return resp.Responses[0].Output, nil
}

// CheckHealth calls the standard grpc.health.v1 Health/Check RPC.
func (c *Client) CheckHealth(ctx context.Context) (*healthpb.HealthCheckResponse, error) {
	return c.healthStub.Check(ctx, &healthpb.HealthCheckRequest{Service: ""})
}

// GetWorkerStatus calls the application-level GetWorkerStatus RPC for detailed status.
func (c *Client) GetWorkerStatus(ctx context.Context) (*pb.WorkerStatusResponse, error) {
	return c.stub.GetWorkerStatus(ctx, &pb.WorkerStatusRequest{})
}

// Close closes the gRPC client connection to the worker.
func (c *Client) Close() error {
	return c.conn.Close()
}
