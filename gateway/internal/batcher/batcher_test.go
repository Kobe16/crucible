package batcher

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"sync"
	"testing"
	"time"

	pb "github.com/Kobe16/crucible/gateway/gen/inference"
)

// fakeInferrer is a test double for the Inferrer interface. It records every
// call. By default, it echoes inputs as outputs ("out-<input>"); set respFn
// to override behavior per-test.
type fakeInferrer struct {
	mu     sync.Mutex
	calls  []*pb.BatchRequest
	respFn func(*pb.BatchRequest) (*pb.BatchResponse, error)
}

func (f *fakeInferrer) BatchInference(_ context.Context, req *pb.BatchRequest) (*pb.BatchResponse, error) {
	f.mu.Lock()
	f.calls = append(f.calls, req)
	f.mu.Unlock()
	if f.respFn != nil {
		return f.respFn(req)
	}
	resps := make([]*pb.InferenceResponse, len(req.Requests))
	for i, r := range req.Requests {
		resps[i] = &pb.InferenceResponse{RequestId: r.RequestId, Output: "out-" + r.Input}
	}
	return &pb.BatchResponse{Responses: resps}, nil
}

func (f *fakeInferrer) callCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.calls)
}

func quietLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

// newPending builds a PendingRequest with a buffered ResponseChan.
func newPending(ctx context.Context, id, input string) *PendingRequest {
	return &PendingRequest{
		RequestID:    id,
		Input:        input,
		ResponseChan: make(chan Result, 1),
		ArrivalTime:  time.Now(),
		Ctx:          ctx,
	}
}

// startBatcher runs b.Run in a goroutine and shuts it down at end of test.
func startBatcher(t *testing.T, b *Batcher) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		b.Run(ctx)
		close(done)
	}()
	t.Cleanup(func() {
		cancel()
		<-done
	})
}

type outcome struct {
	req    *PendingRequest
	result Result
}

// submitAll calls Submit for each request in its own goroutine and returns a
// channel that yields outcomes in completion order (which may differ from
// submission order).
func submitAll(b *Batcher, reqs []*PendingRequest) <-chan outcome {
	out := make(chan outcome, len(reqs))
	for _, r := range reqs {
		go func(r *PendingRequest) { out <- outcome{r, b.Submit(r)} }(r)
	}
	return out
}

// TestBatcher_FlushesOnSize verifies the batcher fires immediately when the
// batch reaches maxBatchSize, without waiting for the timeout.
func TestBatcher_FlushesOnSize(t *testing.T) {
	fake := &fakeInferrer{}
	b := NewBatcher(3, time.Hour, 100, fake, quietLogger())
	startBatcher(t, b)

	reqs := []*PendingRequest{
		newPending(context.Background(), "a", "alpha"),
		newPending(context.Background(), "b", "bravo"),
		newPending(context.Background(), "c", "charlie"),
	}
	results := submitAll(b, reqs)

	for i := 0; i < 3; i++ {
		o := <-results
		if o.result.Err != nil {
			t.Fatalf("unexpected error for %s: %v", o.req.RequestID, o.result.Err)
		}
		want := "out-" + o.req.Input
		if o.result.Output != want {
			t.Errorf("output for %s = %q, want %q", o.req.RequestID, o.result.Output, want)
		}
	}
	if got := fake.callCount(); got != 1 {
		t.Errorf("expected 1 batch call, got %d", got)
	}
}

// TestBatcher_FlushesOnTimeout verifies the batcher fires after batchTimeout
// even when the batch hasn't reached maxBatchSize.
func TestBatcher_FlushesOnTimeout(t *testing.T) {
	fake := &fakeInferrer{}
	timeout := 30 * time.Millisecond
	b := NewBatcher(10, timeout, 100, fake, quietLogger())
	startBatcher(t, b)

	reqs := []*PendingRequest{
		newPending(context.Background(), "a", "alpha"),
		newPending(context.Background(), "b", "bravo"),
	}
	results := submitAll(b, reqs)

	for i := 0; i < 2; i++ {
		o := <-results
		if o.result.Err != nil {
			t.Fatalf("unexpected error for %s: %v", o.req.RequestID, o.result.Err)
		}
		want := "out-" + o.req.Input
		if o.result.Output != want {
			t.Errorf("output for %s = %q, want %q", o.req.RequestID, o.result.Output, want)
		}
	}
	if got := fake.callCount(); got != 1 {
		t.Errorf("expected 1 batch call, got %d", got)
	}
}

// TestBatcher_NoFireOnEmptyQueue verifies that the timer doesn't fire on an
// empty batch — the edge case from issue #24.
func TestBatcher_NoFireOnEmptyQueue(t *testing.T) {
	fake := &fakeInferrer{}
	timeout := 20 * time.Millisecond
	b := NewBatcher(10, timeout, 100, fake, quietLogger())
	startBatcher(t, b)

	// Wait several timeout intervals — if the batcher were going to fire on
	// an empty queue, it would have done so by now.
	time.Sleep(5 * timeout)

	if got := fake.callCount(); got != 0 {
		t.Errorf("expected 0 calls on empty queue, got %d", got)
	}
}

// TestBatcher_DemuxesByRequestID proves the batcher routes responses by
// RequestID rather than slice position. The fake reverses the response order
// before returning; if the batcher relied on order, every output would be
// wrong.
func TestBatcher_DemuxesByRequestID(t *testing.T) {
	fake := &fakeInferrer{
		respFn: func(req *pb.BatchRequest) (*pb.BatchResponse, error) {
			n := len(req.Requests)
			resps := make([]*pb.InferenceResponse, n)
			for i, r := range req.Requests {
				resps[n-1-i] = &pb.InferenceResponse{
					RequestId: r.RequestId,
					Output:    "out-" + r.Input,
				}
			}
			return &pb.BatchResponse{Responses: resps}, nil
		},
	}
	b := NewBatcher(3, time.Hour, 100, fake, quietLogger())
	startBatcher(t, b)

	reqs := []*PendingRequest{
		newPending(context.Background(), "a", "alpha"),
		newPending(context.Background(), "b", "bravo"),
		newPending(context.Background(), "c", "charlie"),
	}
	results := submitAll(b, reqs)

	for i := 0; i < 3; i++ {
		o := <-results
		if o.result.Err != nil {
			t.Fatalf("unexpected error for %s: %v", o.req.RequestID, o.result.Err)
		}
		want := "out-" + o.req.Input
		if o.result.Output != want {
			t.Errorf("RequestID %s: got %q, want %q", o.req.RequestID, o.result.Output, want)
		}
	}
}

// TestBatcher_WorkerErrorFansOut verifies that a gRPC failure unblocks every
// pending request in the batch with the same error.
func TestBatcher_WorkerErrorFansOut(t *testing.T) {
	workerErr := errors.New("worker exploded")
	fake := &fakeInferrer{
		respFn: func(_ *pb.BatchRequest) (*pb.BatchResponse, error) {
			return nil, workerErr
		},
	}
	b := NewBatcher(2, time.Hour, 100, fake, quietLogger())
	startBatcher(t, b)

	reqs := []*PendingRequest{
		newPending(context.Background(), "a", "x"),
		newPending(context.Background(), "b", "y"),
	}
	results := submitAll(b, reqs)

	for i := 0; i < 2; i++ {
		o := <-results
		if !errors.Is(o.result.Err, workerErr) {
			t.Errorf("expected workerErr, got %v", o.result.Err)
		}
	}
}

// TestBatcher_MissingResponseGetsError verifies that requests for which the
// worker didn't return a response don't hang — they're surfaced as errors.
func TestBatcher_MissingResponseGetsError(t *testing.T) {
	fake := &fakeInferrer{
		respFn: func(req *pb.BatchRequest) (*pb.BatchResponse, error) {
			// Return responses only for the first two of three requests.
			return &pb.BatchResponse{
				Responses: []*pb.InferenceResponse{
					{RequestId: req.Requests[0].RequestId, Output: "out-0"},
					{RequestId: req.Requests[1].RequestId, Output: "out-1"},
				},
			}, nil
		},
	}
	b := NewBatcher(3, time.Hour, 100, fake, quietLogger())
	startBatcher(t, b)

	reqs := []*PendingRequest{
		newPending(context.Background(), "a", "x"),
		newPending(context.Background(), "b", "y"),
		newPending(context.Background(), "c", "z"),
	}
	results := submitAll(b, reqs)

	// Which specific RequestID lands in the unresponded slot depends on the
	// (racy) order in which the three goroutines reach the queue, so just
	// verify counts: two requests succeed, one gets an error.
	var ok, errCount int
	for i := 0; i < 3; i++ {
		o := <-results
		if o.result.Err != nil {
			errCount++
		} else {
			ok++
		}
	}
	if ok != 2 || errCount != 1 {
		t.Errorf("ok=%d err=%d, want ok=2 err=1", ok, errCount)
	}
}

// TestBatcher_SubmitReturnsCtxErrOnCancel verifies Submit returns the
// request's ctx error when the request is cancelled before a result arrives.
func TestBatcher_SubmitReturnsCtxErrOnCancel(t *testing.T) {
	// Slow fake — won't respond before the request ctx deadlines out.
	fake := &fakeInferrer{
		respFn: func(req *pb.BatchRequest) (*pb.BatchResponse, error) {
			time.Sleep(100 * time.Millisecond)
			resps := make([]*pb.InferenceResponse, len(req.Requests))
			for i, r := range req.Requests {
				resps[i] = &pb.InferenceResponse{RequestId: r.RequestId, Output: "ignored"}
			}
			return &pb.BatchResponse{Responses: resps}, nil
		},
	}
	b := NewBatcher(1, time.Hour, 100, fake, quietLogger())
	startBatcher(t, b)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	req := newPending(ctx, "a", "x")

	result := b.Submit(req)
	if !errors.Is(result.Err, context.DeadlineExceeded) {
		t.Errorf("expected DeadlineExceeded, got %v", result.Err)
	}
}
