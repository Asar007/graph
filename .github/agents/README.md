# Custom Agents

This directory contains custom agent definitions for GitHub Copilot to assist with specific tasks in this repository.

## Available Agents

### sequence-diagram.md

A specialized agent for creating sequence diagrams. This agent understands:
- The JSON schema used by the graph visualization app
- UML sequence diagram conventions
- How to structure participants, events, activations, and fragments

**Use this agent when:**
- Creating new sequence diagrams for API flows
- Designing system interaction diagrams
- Documenting authentication/authorization flows
- Visualizing request/response patterns

**Example prompt:**
```
Create a sequence diagram for user authentication with OAuth2
```

## How to Use

These agents are automatically available when using GitHub Copilot in this repository. Simply describe the type of diagram you want to create, and the agent will generate the appropriate JSON structure that can be used with the visualization application.
