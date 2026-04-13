package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/Kobe16/crucible/gateway/internal/config"
	"github.com/Kobe16/crucible/gateway/internal/handler"
	"github.com/Kobe16/crucible/gateway/internal/worker"
)

// Main entry point for the gateway application. Initializes configuration, sets up gRPC client and HTTP server, and handles graceful shutdown.
func main() {
	cfg := config.Load()

	client, err := worker.NewClient(cfg.WorkerAddr)
	if err != nil {
		log.Fatalf("Failed to create gRPC client: %v", err)
	}

	// Startup health check — log but don't crash if worker isn't ready yet
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	resp, err := client.CheckHealth(ctx)
	cancel()
	if err != nil {
		log.Printf("WARNING: worker health check failed: %v", err)
	} else {
		log.Printf("Worker health: %s", resp.Status)
	}

	// Setup HTTP server with Router (mux) that dispatches to handler functions for each endpoint
	h := handler.New(client)

	mux := http.NewServeMux()
	mux.HandleFunc("POST /predict", h.Predict)
	mux.HandleFunc("GET /health", h.Health)
	mux.HandleFunc("GET /status", h.Status)

	server := &http.Server{
		Addr:    ":" + cfg.HTTPPort,
		Handler: mux,
	}

	// Start HTTP server is a goroutine so we can listen for shutdown signals in the main thread
	go func() {
		log.Printf("HTTP server listening on :%s", cfg.HTTPPort)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("HTTP server error: %v", err)
		}
	}()

	// In main thread, listen for OS signals to gracefully shutdown the server and gRPC client
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGTERM, syscall.SIGINT)	// SIGTERM for Kubernetes, SIGINT for Ctrl+C
	sig := <-sigCh	// Pause here till we receive a shutdown signal
	log.Printf("Received %s, shutting down", sig)

	// Create context with timeout for graceful shutdown of HTTP server and gRPC client
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer shutdownCancel()

	if err := server.Shutdown(shutdownCtx); err != nil {
		log.Printf("HTTP shutdown error: %v", err)
	}

	if err := client.Close(); err != nil {
		log.Printf("gRPC close error: %v", err)
	}

	log.Println("Gateway stopped")
}