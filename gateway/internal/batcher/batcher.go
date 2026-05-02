package batcher

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	pb "github.com/Kobe16/crucible/gateway/gen/inference"
)

// Inferrer is the narrow worker-facing dependency the batcher needs.
// inference.Client satisfies this interface.
type Inferrer interface {
	BatchInference(ctx context.Context, req *pb.BatchRequest) (*pb.BatchResponse, error)
}

// Batcher collects PendingRequests off a queue and dispatches them to the
// worker in batches, firing on whichever happens first: the batch reaching
// maxBatchSize, or batchTimeout elapsing since the first request of the
// current batch arrived.
type Batcher struct {
	queue        chan *PendingRequest
	maxBatchSize int
	batchTimeout time.Duration
	inferrer     Inferrer
	logger       *slog.Logger
}

// NewBatcher constructs a Batcher with a buffered queue sized to queueDepth.
// Run must be called (typically in a goroutine) to start consuming the queue.
func NewBatcher(maxBatchSize int, batchTimeout time.Duration, queueDepth int, inferrer Inferrer, logger *slog.Logger) *Batcher {
	return &Batcher{
		queue:        make(chan *PendingRequest, queueDepth),
		maxBatchSize: maxBatchSize,
		batchTimeout: batchTimeout,
		inferrer:     inferrer,
		logger:       logger,
	}
}

// Submit enqueues a PendingRequest and blocks until the batcher returns a
// Result or the request's context is cancelled.
func (b *Batcher) Submit(req *PendingRequest) Result {
	// Don't enqueue request if its context is already cancelled (from 10s handler timeout, or client disconnect)
	select {
	case b.queue <- req:
	case <-req.Ctx.Done():
		return Result{Err: req.Ctx.Err()}
	}

	select {
	case result := <-req.ResponseChan:
		return result

	case <-req.Ctx.Done():
		return Result{Err: req.Ctx.Err()}
	}
}

// Run is the batcher's goroutine loop. It selects on the queue and a per-batch
// timer, flushing whenever the batch hits maxBatchSize or the timer fires.
// Returns when ctx is cancelled.
func (b *Batcher) Run(ctx context.Context) {
	var batch []*PendingRequest
	var timer *time.Timer
	var timerC <-chan time.Time // nil when no timer is running

	for {
		select {
		case <-ctx.Done():
			return

		case req := <-b.queue:
			batch = append(batch, req)

			// Start a timer on the first request of a new batch
			if len(batch) == 1 {
				timer = time.NewTimer(b.batchTimeout)
				timerC = timer.C
			}

			// Flush on size
			if len(batch) >= b.maxBatchSize {
				// stop the timer; if it already fired, drain the buffered tick so it doesn't leak into the next batch
				if !timer.Stop() {
					select {
					case <-timer.C:
					default:
					}
				}

				go b.flush(ctx, batch)
				batch = nil
				timer = nil
				timerC = nil
			}

		// Flush on timeout
		case <-timerC:
			go b.flush(ctx, batch)
			batch = nil
			timer = nil
			timerC = nil
		}
	}
}

// flush builds a BatchRequest from the collected PendingRequests, sends it to
// the worker, then demuxes the BatchResponse back to each request's
// ResponseChan by request_id.
func (b *Batcher) flush(ctx context.Context, batch []*PendingRequest) {
	requests := make([]*pb.InferenceRequest, len(batch))
	for i, req := range batch {
		requests[i] = &pb.InferenceRequest{
			RequestId:  req.RequestID,
			Input:      req.Input,
			Parameters: req.Parameters,
		}
	}

	resp, err := b.inferrer.BatchInference(ctx, &pb.BatchRequest{Requests: requests})
	if err != nil {
		for _, req := range batch {
			req.ResponseChan <- Result{Err: err}
		}
		return
	}

	// Build map of pending requests keyed by RequestID for fast lookup
	pendingByID := make(map[string]*PendingRequest, len(batch))
	for _, req := range batch {
		pendingByID[req.RequestID] = req
	}

	// Delete requests from map as you return their responses
	for _, response := range resp.Responses {
		if req, ok := pendingByID[response.RequestId]; ok {
			req.ResponseChan <- Result{Output: response.Output}
			delete(pendingByID, response.RequestId)
		} else {
			b.logger.Warn("unknown_response_id", "request_id", response.RequestId)
		}
	}

	// For requests that didn't get a response, send an error
	for _, req := range pendingByID {
		req.ResponseChan <- Result{
			Err: fmt.Errorf("worker did not return a response for request %s", req.RequestID),
		}
	}
}
