package batcher

import (
	"context"
	"errors"
	"time"
)

// ErrQueueFull is returned by Submit when the batcher queue is at capacity.
var ErrQueueFull = errors.New("batcher queue full")

// PendingRequest is a single inference request waiting in the batcher queue.
// The handler creates one per HTTP request, hands it to Submit, then blocks on
// ResponseChan until the batcher fills it in.
type PendingRequest struct {
	RequestID    string
	Input        string
	Parameters   map[string]string
	ResponseChan chan Result
	ArrivalTime  time.Time
	Ctx          context.Context // for detecting client disconnect
}

// Result carries either a successful inference output or an error back to the
// handler that submitted the request.
type Result struct {
	Output string
	Err    error
}
