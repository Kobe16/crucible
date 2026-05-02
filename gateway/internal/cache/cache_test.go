package cache

import (
	"testing"
	"time"
)

func TestNoopCache_AlwaysMisses(t *testing.T) {
	c := &NoopCache{}
	c.Set("key", "value", time.Minute)
	if _, ok := c.Get("key"); ok {
		t.Error("NoopCache.Get returned ok=true, want false")
	}
}

func TestCacheKey_Deterministic(t *testing.T) {
	params := map[string]string{"temp": "0.7", "top_k": "50"}
	k1 := CacheKey("hello", params)
	k2 := CacheKey("hello", params)
	if k1 != k2 {
		t.Errorf("same input produced different keys: %q vs %q", k1, k2)
	}
}

func TestCacheKey_DifferentInput(t *testing.T) {
	params := map[string]string{"temp": "0.7"}
	k1 := CacheKey("hello", params)
	k2 := CacheKey("world", params)
	if k1 == k2 {
		t.Error("different inputs produced same key")
	}
}

func TestCacheKey_DifferentParams(t *testing.T) {
	k1 := CacheKey("hello", map[string]string{"temp": "0.7"})
	k2 := CacheKey("hello", map[string]string{"temp": "0.9"})
	if k1 == k2 {
		t.Error("different params produced same key")
	}
}

func TestCacheKey_ParamOrderIrrelevant(t *testing.T) {
	k1 := CacheKey("hello", map[string]string{"a": "1", "b": "2"})
	k2 := CacheKey("hello", map[string]string{"b": "2", "a": "1"})
	if k1 != k2 {
		t.Errorf("param order changed key: %q vs %q", k1, k2)
	}
}

func TestCacheKey_NilParams(t *testing.T) {
	k1 := CacheKey("hello", nil)
	k2 := CacheKey("hello", map[string]string{})
	if k1 != k2 {
		t.Errorf("nil vs empty params produced different keys: %q vs %q", k1, k2)
	}
}

func TestCacheKey_HexLength(t *testing.T) {
	k := CacheKey("test", nil)
	if len(k) != 64 {
		t.Errorf("expected 64-char hex SHA-256, got %d chars: %q", len(k), k)
	}
}
