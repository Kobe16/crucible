package config

import (
	"fmt"
	"os"
	"strconv"
	"time"
)

// Config holds environment variables to start the application.
type Config struct {
	WorkerAddr        string
	HTTPPort          string
	LogLevel          string
	MaxBatchSize      int
	BatchTimeout      time.Duration
	InferenceDeadline time.Duration
	QueueDepth        int
}

// Load reads environment variables and returns a populated Config struct.
// Returns an error if any numeric value would produce invalid batcher behaviour.
func Load() (Config, error) {
	maxBatchSize := getEnv("MAX_BATCH_SIZE", 8)
	batchTimeoutMS := getEnv("BATCH_TIMEOUT_MS", 50)
	inferenceDeadlineMS := getEnv("INFERENCE_DEADLINE_MS", 2000)
	queueDepth := getEnv("QUEUE_DEPTH", 1000)

	if maxBatchSize <= 0 {
		return Config{}, fmt.Errorf("MAX_BATCH_SIZE must be positive, got %d", maxBatchSize)
	}
	if batchTimeoutMS <= 0 {
		return Config{}, fmt.Errorf("BATCH_TIMEOUT_MS must be positive, got %d", batchTimeoutMS)
	}
	if inferenceDeadlineMS <= 0 {
		return Config{}, fmt.Errorf("INFERENCE_DEADLINE_MS must be positive, got %d", inferenceDeadlineMS)
	}
	if queueDepth <= 0 {
		return Config{}, fmt.Errorf("QUEUE_DEPTH must be positive, got %d", queueDepth)
	}

	return Config{
		WorkerAddr:        getEnv("WORKER_ADDR", "worker:50051"),
		HTTPPort:          getEnv("HTTP_PORT", "8080"),
		LogLevel:          getEnv("LOG_LEVEL", "info"),
		MaxBatchSize:      maxBatchSize,
		BatchTimeout:      time.Duration(batchTimeoutMS) * time.Millisecond,
		InferenceDeadline: time.Duration(inferenceDeadlineMS) * time.Millisecond,
		QueueDepth:        queueDepth,
	}, nil
}

// getEnv returns the value of the environment variable specified by key,
// parsed as T. Invalid int values fall back to the default.
func getEnv[T string | int](key string, fallback T) T {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	switch any(fallback).(type) {
	case int:
		if n, err := strconv.Atoi(v); err == nil {
			return any(n).(T)
		}
		return fallback
	default:
		return any(v).(T)
	}
}
