package config

import "os"

// Config holds environment variables to start the application.
type Config struct {
	WorkerAddr string
	HTTPPort   string
	LogLevel   string
}

// Load reads environment variables and returns a populated Config struct.
func Load() Config {
	return Config{
		WorkerAddr: getEnv("WORKER_ADDR", "worker:50051"),
		HTTPPort:   getEnv("HTTP_PORT", "8080"),
		LogLevel:   getEnv("LOG_LEVEL", "info"),
	}
}

// getEnv returns the value of the environment variable specified by key.
func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
