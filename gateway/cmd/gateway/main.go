package main

import (
	"context"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/Kobe16/crucible/gateway/internal/config"
	"github.com/Kobe16/crucible/gateway/internal/handler"
	"github.com/Kobe16/crucible/gateway/internal/inference"
)

// Main entry point for the gateway application. Initializes configuration, sets up gRPC client and HTTP server, and handles graceful shutdown.
func main() {
	cfg := config.Load()

	// Initialize slog with JSON output. Level defaults to INFO if LOG_LEVEL is unrecognized.
	var level slog.Level
	if err := level.UnmarshalText([]byte(cfg.LogLevel)); err != nil {
		level = slog.LevelInfo
	}
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: level})))

	client, err := inference.NewClient(cfg.WorkerAddr)
	if err != nil {
		slog.Error("grpc_client_init_failed", "error", err)
		os.Exit(1)
	}

	// Startup health check — log but don't crash if worker isn't ready yet
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	resp, err := client.CheckHealth(ctx)
	cancel()
	if err != nil {
		slog.Warn("worker_health_check_failed", "error", err)
	} else {
		slog.Info("worker_health_check", "status", resp.Status)
	}

	// Setup HTTP server with Router (mux) that dispatches to handler functions for each endpoint
	// TODO(step F): construct the real batcher and pass it here.
	h := handler.New(client, nil)

	mux := http.NewServeMux()
	mux.HandleFunc("POST /predict", h.Predict)
	mux.HandleFunc("GET /health", h.Health)
	mux.HandleFunc("GET /status", h.Status)

	server := &http.Server{
		Addr:    ":" + cfg.HTTPPort,
		Handler: handler.LoggingMiddleware(mux),
	}

	// Start HTTP server is a goroutine so we can listen for shutdown signals in the main thread
	go func() {
		slog.Info("http_server_listening", "port", cfg.HTTPPort)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			slog.Error("http_server_error", "error", err)
			os.Exit(1)
		}
	}()

	// In main thread, listen for OS signals to gracefully shutdown the server and gRPC client
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGTERM, syscall.SIGINT) // SIGTERM for Kubernetes, SIGINT for Ctrl+C
	sig := <-sigCh                                        // Pause here till we receive a shutdown signal
	slog.Info("shutdown_signal", "signal", sig.String())

	// Create context with timeout for graceful shutdown of HTTP server and gRPC client
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer shutdownCancel()

	if err := server.Shutdown(shutdownCtx); err != nil {
		slog.Error("http_shutdown_error", "error", err)
	}

	if err := client.Close(); err != nil {
		slog.Error("grpc_close_error", "error", err)
	}

	slog.Info("gateway_stopped")
}
