package trusera

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// EnforcementAction defines how policy violations are handled
type EnforcementAction string

const (
	EnforcementLog   EnforcementAction = "log"
	EnforcementWarn  EnforcementAction = "warn"
	EnforcementBlock EnforcementAction = "block"
)

// StandaloneInterceptor intercepts HTTP requests and evaluates them against Cedar policies
type StandaloneInterceptor struct {
	policyFile      string
	enforcement     EnforcementAction
	logFile         string
	excludePatterns []string
	rules           []PolicyRule
	logMu           sync.Mutex
	logWriter       *os.File
}

// StandaloneOption configures a StandaloneInterceptor
type StandaloneOption func(*StandaloneInterceptor)

// WithPolicyFile sets the path to the Cedar policy file
func WithPolicyFile(p string) StandaloneOption {
	return func(si *StandaloneInterceptor) {
		si.policyFile = filepath.Clean(p)
	}
}

// WithEnforcement sets the enforcement mode (log, warn, block)
func WithEnforcement(mode EnforcementAction) StandaloneOption {
	return func(si *StandaloneInterceptor) {
		si.enforcement = mode
	}
}

// WithLogFile sets the path to the JSONL event log file
func WithLogFile(p string) StandaloneOption {
	return func(si *StandaloneInterceptor) {
		si.logFile = filepath.Clean(p)
	}
}

// WithExcludePatterns sets URL patterns to skip interception
func WithExcludePatterns(patterns ...string) StandaloneOption {
	return func(si *StandaloneInterceptor) {
		si.excludePatterns = patterns
	}
}

// NewStandaloneInterceptor creates a standalone interceptor with Cedar policy evaluation
func NewStandaloneInterceptor(opts ...StandaloneOption) (*StandaloneInterceptor, error) {
	si := &StandaloneInterceptor{
		enforcement:     EnforcementLog,
		excludePatterns: []string{},
	}

	for _, opt := range opts {
		opt(si)
	}

	// Load policy file if specified
	if si.policyFile != "" {
		content, err := os.ReadFile(si.policyFile)
		if err != nil {
			return nil, fmt.Errorf("failed to read policy file: %w", err)
		}

		rules, err := ParseCedarPolicy(string(content))
		if err != nil {
			return nil, fmt.Errorf("failed to parse policy: %w", err)
		}

		si.rules = rules
	}

	// Open log file if specified
	if si.logFile != "" {
		f, err := os.OpenFile(si.logFile, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0600)
		if err != nil {
			return nil, fmt.Errorf("failed to open log file: %w", err)
		}
		si.logWriter = f
	}

	return si, nil
}

// WrapClient wraps an http.Client to intercept requests
func (si *StandaloneInterceptor) WrapClient(client *http.Client) *http.Client {
	if client == nil {
		client = &http.Client{}
	}

	transport := client.Transport
	if transport == nil {
		transport = http.DefaultTransport
	}

	client.Transport = &standaloneTransport{
		base:        transport,
		interceptor: si,
	}

	return client
}

// Close flushes and closes the log file
func (si *StandaloneInterceptor) Close() error {
	si.logMu.Lock()
	defer si.logMu.Unlock()

	if si.logWriter != nil {
		return si.logWriter.Close()
	}

	return nil
}

// standaloneTransport implements http.RoundTripper
type standaloneTransport struct {
	base        http.RoundTripper
	interceptor *StandaloneInterceptor
}

// eventLog represents a JSONL log entry
type eventLog struct {
	Timestamp         string  `json:"timestamp"`
	Method            string  `json:"method"`
	URL               string  `json:"url"`
	Hostname          string  `json:"hostname"`
	Path              string  `json:"path"`
	Status            int     `json:"status,omitempty"`
	DurationMs        float64 `json:"duration_ms"`
	PolicyDecision    string  `json:"policy_decision"`
	EnforcementAction string  `json:"enforcement_action"`
	Reasons           string  `json:"reasons,omitempty"`
}

// RoundTrip intercepts HTTP requests and evaluates Cedar policies
func (t *standaloneTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	// Check if URL should be excluded
	if t.shouldExclude(req.URL.String()) {
		return t.base.RoundTrip(req)
	}

	startTime := time.Now()

	// Build request context
	ctx := RequestContext{
		URL:      req.URL.String(),
		Method:   req.Method,
		Hostname: req.URL.Hostname(),
		Path:     req.URL.Path,
	}

	// Evaluate policy
	decision := EvaluatePolicy(ctx, t.interceptor.rules)

	// Determine enforcement action
	var enforcementAction string
	var blockRequest bool

	if decision.Decision == "Deny" {
		switch t.interceptor.enforcement {
		case EnforcementBlock:
			enforcementAction = "blocked"
			blockRequest = true
		case EnforcementWarn:
			enforcementAction = "warned"
			blockRequest = false
		case EnforcementLog:
			enforcementAction = "logged"
			blockRequest = false
		}
	} else {
		enforcementAction = "allowed"
		blockRequest = false
	}

	// Handle blocking
	if blockRequest {
		duration := time.Since(startTime).Milliseconds()
		t.logEvent(eventLog{
			Timestamp:         time.Now().UTC().Format(time.RFC3339),
			Method:            req.Method,
			URL:               req.URL.String(),
			Hostname:          req.URL.Hostname(),
			Path:              req.URL.Path,
			DurationMs:        float64(duration),
			PolicyDecision:    decision.Decision,
			EnforcementAction: enforcementAction,
			Reasons:           strings.Join(decision.Reasons, "; "),
		})

		return nil, fmt.Errorf("request blocked by Cedar policy: %s", strings.Join(decision.Reasons, "; "))
	}

	// Forward request
	resp, err := t.base.RoundTrip(req)

	duration := time.Since(startTime).Milliseconds()

	// Log event
	logEntry := eventLog{
		Timestamp:         time.Now().UTC().Format(time.RFC3339),
		Method:            req.Method,
		URL:               req.URL.String(),
		Hostname:          req.URL.Hostname(),
		Path:              req.URL.Path,
		DurationMs:        float64(duration),
		PolicyDecision:    decision.Decision,
		EnforcementAction: enforcementAction,
	}

	if len(decision.Reasons) > 0 {
		logEntry.Reasons = strings.Join(decision.Reasons, "; ")
	}

	if resp != nil {
		logEntry.Status = resp.StatusCode
	}

	t.logEvent(logEntry)

	return resp, err
}

// shouldExclude checks if URL matches any exclude patterns
func (t *standaloneTransport) shouldExclude(urlStr string) bool {
	for _, pattern := range t.interceptor.excludePatterns {
		// Support both substring match and regex-like patterns
		if strings.Contains(urlStr, pattern) {
			return true
		}
	}
	return false
}

// logEvent writes an event to the JSONL log file
func (t *standaloneTransport) logEvent(entry eventLog) {
	if t.interceptor.logWriter == nil {
		return
	}

	t.interceptor.logMu.Lock()
	defer t.interceptor.logMu.Unlock()

	data, err := json.Marshal(entry)
	if err != nil {
		return
	}

	data = append(data, '\n')
	t.interceptor.logWriter.Write(data)
}

// MustNewStandaloneInterceptor creates a standalone interceptor or panics on error
func MustNewStandaloneInterceptor(opts ...StandaloneOption) *StandaloneInterceptor {
	si, err := NewStandaloneInterceptor(opts...)
	if err != nil {
		panic(fmt.Sprintf("failed to create standalone interceptor: %v", err))
	}
	return si
}

// ParseURL is a helper to extract hostname and path from a URL string
func ParseURL(rawURL string) (hostname, path string) {
	u, err := url.Parse(rawURL)
	if err != nil {
		return "", ""
	}
	return u.Hostname(), u.Path
}
