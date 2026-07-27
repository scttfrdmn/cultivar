// Command cultivar answers whether to self-host a Hugging Face model on AWS or
// use Bedrock serverless, and then deploys the answer.
//
// The commands land over the v0.1–v0.2 milestones; this entry point exists so the
// module builds and CI has a target from the first commit.
package main

import (
	"fmt"
	"os"
)

// Version is set at build time via -ldflags (see Makefile).
var Version = "0.0.0-dev"

func main() {
	if len(os.Args) > 1 {
		switch os.Args[1] {
		case "version", "--version", "-v":
			fmt.Printf("cultivar %s\n", Version)
			return
		}
	}
	fmt.Fprintf(os.Stderr, "cultivar %s — no commands implemented yet.\n", Version)
	fmt.Fprintln(os.Stderr, "See https://github.com/scttfrdmn/cultivar/milestones")
	os.Exit(1)
}
