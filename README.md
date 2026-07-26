# Clinical Trials Query-to-Visualization Agent

Backend service that converts natural-language clinical-trial questions into structured
visualization specifications backed by the ClinicalTrials.gov API.

## Status

Scaffold in progress — see `BUILD_PLAN_FINAL.md` (not tracked in this repo) for the full
phase-by-phase implementation plan.

## Quick Start

_TODO: filled in as phases land._

## How It Works

Two-touch agent architecture: an LLM plans/classifies the request, a fully deterministic
pipeline fetches and aggregates ClinicalTrials.gov data, and a second LLM touch interprets
the results. See design docs (added in later phases) for details.

## API Reference

_TODO_

## Design Decisions

_TODO_

## AI Tools & Integrity

_TODO_

## Limitations

_TODO_
