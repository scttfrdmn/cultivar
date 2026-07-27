.PHONY: all build install test test-live check fmt vet lint clean help

BINARY_NAME=cultivar
VERSION?=0.0.0-dev
BUILD_DIR=bin
MAIN_PATH=./cmd/cultivar

GOCMD=go
LDFLAGS=-ldflags "-X main.Version=$(VERSION) -s -w"

all: check build

## build: Build the binary
build:
	@mkdir -p $(BUILD_DIR)
	$(GOCMD) build $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME) $(MAIN_PATH)
	@echo "Built $(BUILD_DIR)/$(BINARY_NAME)"

## install: Install to GOBIN
install:
	$(GOCMD) install $(LDFLAGS) $(MAIN_PATH)

## test: Run tests (no credentials required)
test:
	$(GOCMD) test ./...

## test-live: Run tests that hit the free read-only AWS APIs (needs credentials).
## Catches AWS schema drift. Never launches billable resources.
test-live:
	$(GOCMD) test -tags live ./...

## fmt: Check formatting
fmt:
	@gofmt -l . | tee /tmp/cultivar-fmt && test ! -s /tmp/cultivar-fmt

## vet: Run go vet
vet:
	$(GOCMD) vet ./...

## check: build + vet + fmt + test — run this before opening a PR
check: build vet fmt test

## clean: Remove build artifacts
clean:
	$(GOCMD) clean
	@rm -rf $(BUILD_DIR)

## help: Show available targets
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## //'
