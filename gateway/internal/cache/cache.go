package cache

import (
	"crypto/sha256"
	"encoding/hex"
	"sort"
	"time"
)

// Cache is the lookup interface for inference result caching.
// The initial implementation is NoopCache (always a miss, zero overhead).
// To add Redis: implement RedisCache satisfying this interface and inject via config.
type Cache interface {
	Get(key string) (string, bool)
	Set(key string, value string, ttl time.Duration)
}

// NoopCache is the default no-op implementation — always a cache miss.
type NoopCache struct{}

func (c *NoopCache) Get(_ string) (string, bool)             { return "", false }
func (c *NoopCache) Set(_ string, _ string, _ time.Duration) {}

// CacheKey returns a deterministic cache key for the given input and parameters.
// Key: sha256(input + sorted parameters) encoded as hex.
func CacheKey(input string, params map[string]string) string {
	keys := make([]string, 0, len(params))
	for k := range params {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	h := sha256.New()
	h.Write([]byte(input))
	for _, k := range keys {
		h.Write([]byte(k + "=" + params[k]))
	}
	return hex.EncodeToString(h.Sum(nil))
}
