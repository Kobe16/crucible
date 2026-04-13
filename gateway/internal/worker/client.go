package worker

import (
	"context"
	"fmt"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/Kobe16/crucible/gateway/gen/inference"
)

// Client holds the gRPC client connection and stub for communicating with the worker.
type Client struct {
	conn *grpc.ClientConn
	stub pb.InferenceServiceClient
}

// NewClient creates a new gRPC client for the worker at the specified address.
func NewClient(addr string) (*Client, error) {
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("grpc new client: %w", err)
	}
	return &Client{
		conn: conn,
		stub: pb.NewInferenceServiceClient(conn),
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

// HealthCheck sends a health check request to the worker and returns the response.
func (c *Client) HealthCheck(ctx context.Context) (*pb.HealthResponse, error) {
	return c.stub.HealthCheck(ctx, &pb.HealthRequest{})
}

// Close closes the gRPC client connection to the worker.
func (c *Client) Close() error {
	return c.conn.Close()
}