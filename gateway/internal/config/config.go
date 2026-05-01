package config

import (
	"os"
	"strconv"
	"time"
)

// Config holds environment variables to start the application.
type Config struct {
	WorkerAddr   string
	HTTPPort     string
	LogLevel     string
	MaxBatchSize int
	BatchTimeout time.Duration
	QueueDepth   int
}

// Load reads environment variables and returns a populated Config struct.
func Load() Config {
	return Config{
		WorkerAddr:   getEnv("WORKER_ADDR", "worker:50051"),
		HTTPPort:     getEnv("HTTP_PORT", "8080"),
		LogLevel:     getEnv("LOG_LEVEL", "info"),
		MaxBatchSize: getEnv("MAX_BATCH_SIZE", 8),
		BatchTimeout: time.Duration(getEnv("BATCH_TIMEOUT_MS", 50)) * time.Millisecond,
		QueueDepth:   getEnv("QUEUE_DEPTH", 1000),
	}
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
