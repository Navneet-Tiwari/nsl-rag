"""
ecommerce.py
------------
Synthetic e-commerce microservices dataset for NSL-RAG demonstration.

Defines a realistic e-commerce system with:
- 9 services
- 3 databases
- 3 deployment events
- 3 intentional failure scenarios designed for multi-hop reasoning

This dataset is the foundation for all demos and benchmarks.
The lattice structure mirrors real-world microservice dependencies.

Usage:
    from nsl_rag.data.ecommerce import EcommerceSystem

    raw_nodes = EcommerceSystem.get_raw_nodes()
    scenarios = EcommerceSystem.get_failure_scenarios()
"""

from nsl_rag.core.logger import get_logger

log = get_logger(__name__)


class EcommerceSystem:
    """
     Synthetic e-commerce microservices system.

     Architecture:
                         [API Gateway]
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                  │
    [Order Service]   [Payment Service]   [User Service]
           │                  │                  │
           │         [Fraud Detection]      [User DB]
           │                  │
    [Inventory]         [Payment DB]
           │
    [Warehouse]
           │
    [Notification]
           │
    [Email Service]
    """

    @staticmethod
    def get_raw_nodes() -> list[dict]:
        """
        Returns all raw node definitions for the e-commerce system.
        Pass directly to LatticeBuilder.build() to construct the lattice.
        """
        return [
            *EcommerceSystem._root_nodes(),
            *EcommerceSystem._service_nodes(),
            *EcommerceSystem._database_nodes(),
            *EcommerceSystem._deployment_nodes(),
        ]

    @staticmethod
    def get_failure_scenarios() -> list[dict]:
        """
        Returns the three failure scenarios used for benchmarking.
        Each scenario has a query and expected reasoning trace.
        """
        return [
            {
                "id": "scenario_1",
                "name": "The Classic Cascade",
                "query": "Why is the payment service failing?",
                "expected_tags": ["payment", "critical"],
                "expected_nodes": [
                    "payment_service",
                    "fraud_detection",
                    "payment_db",
                    "flash_sale_deployment",
                ],
                "hop_count": 3,
                "description": (
                    "Flash Sale deployment pushed 10x traffic. "
                    "Payment Service overloaded. "
                    "Fraud Detection timeout. "
                    "Orders failing at checkout."
                ),
            },
            {
                "id": "scenario_2",
                "name": "The Silent Dependency",
                "query": "Why is the payment database returning errors?",
                "expected_tags": ["payment", "database"],
                "expected_nodes": [
                    "payment_db",
                    "payment_service",
                    "fraud_model_deployment",
                ],
                "hop_count": 4,
                "description": (
                    "Fraud Model v3 deployment increased CPU usage. "
                    "Payment DB connection pool exhausted. "
                    "Payment Service returning 503."
                ),
            },
            {
                "id": "scenario_3",
                "name": "The Innocent Bystander",
                "query": "Why are orders stuck in pending state?",
                "expected_tags": ["orders", "critical"],
                "expected_nodes": [
                    "order_service",
                    "notification_service",
                    "email_service",
                ],
                "hop_count": 5,
                "description": (
                    "Email Service crashed. "
                    "Notification Service timeout. "
                    "Order Service waiting for confirmation. "
                    "Orders stuck in pending."
                ),
            },
        ]

    # ── Node Definitions ──────────────────────────────────────────────────────

    @staticmethod
    def _root_nodes() -> list[dict]:
        return [
            {
                "node_id": "ecommerce_root",
                "node_type": "root",
                "title": "E-Commerce System",
                "summary": "Root node of the e-commerce microservices lattice",
                "content": (
                    "Top-level system encompassing all microservices, "
                    "databases, and deployment events for the e-commerce platform."
                ),
                "tags": ["root", "system", "ecommerce"],
                "children": [
                    "api_gateway",
                ],
                "parents": [],
                "metadata": {"version": "2.1.0", "environment": "production"},
            },
        ]

    @staticmethod
    def _service_nodes() -> list[dict]:
        return [
            # ── API Gateway ───────────────────────────────────────────────────
            {
                "node_id": "api_gateway",
                "node_type": "service",
                "title": "API Gateway",
                "summary": "Entry point for all incoming traffic",
                "content": (
                    "API Gateway routes all external requests to internal services. "
                    "Runs on port 443. Handles SSL termination, rate limiting, "
                    "and request routing. All services depend on it being healthy."
                ),
                "tags": ["gateway", "service", "critical", "infrastructure"],
                "children": ["order_service", "payment_service", "user_service"],
                "parents": ["ecommerce_root"],
                "metadata": {"port": 443, "owner": "platform-team"},
            },
            # ── Order Service ─────────────────────────────────────────────────
            {
                "node_id": "order_service",
                "node_type": "service",
                "title": "Order Service",
                "summary": "Creates and manages customer orders",
                "content": (
                    "Order Service handles order creation, status updates, "
                    "and order history. Runs on port 8081. "
                    "Depends on Inventory Service for stock validation "
                    "and Notification Service for order confirmations. "
                    "Orders remain in PENDING state if Notification Service is unavailable."
                ),
                "tags": ["orders", "service", "critical"],
                "children": ["inventory_service", "notification_service", "orders_db"],
                "parents": ["api_gateway"],
                "metadata": {"port": 8081, "owner": "orders-team"},
            },
            # ── Payment Service ───────────────────────────────────────────────
            {
                "node_id": "payment_service",
                "node_type": "service",
                "title": "Payment Service",
                "summary": "Handles all payment processing for the platform",
                "content": (
                    "Payment Service processes all customer payments. "
                    "Runs on port 8080. "
                    "Every payment must pass Fraud Detection before processing. "
                    "Depends on Payment DB for transaction storage. "
                    "Returns 503 if Fraud Detection times out or Payment DB "
                    "connection pool is exhausted. "
                    "SLA: 99.99% uptime. Max response time: 2 seconds."
                ),
                "tags": ["payment", "service", "critical", "database"],
                "children": ["fraud_detection", "payment_db"],
                "parents": ["api_gateway"],
                "metadata": {
                    "port": 8080,
                    "owner": "payments-team",
                    "sla": "99.99%",
                },
            },
            # ── User Service ──────────────────────────────────────────────────
            {
                "node_id": "user_service",
                "node_type": "service",
                "title": "User Service",
                "summary": "Manages user accounts and authentication",
                "content": (
                    "User Service handles registration, login, and profile management. "
                    "Runs on port 8082. "
                    "Depends on User DB for all user data. "
                    "Not on the critical payment path."
                ),
                "tags": ["users", "service", "authentication"],
                "children": ["user_db"],
                "parents": ["api_gateway"],
                "metadata": {"port": 8082, "owner": "identity-team"},
            },
            # ── Fraud Detection ───────────────────────────────────────────────
            {
                "node_id": "fraud_detection",
                "node_type": "service",
                "title": "Fraud Detection Service",
                "summary": "Validates payment legitimacy before processing",
                "content": (
                    "Fraud Detection uses ML scoring to validate each payment. "
                    "Timeout threshold: 2 seconds. "
                    "If Fraud Detection times out, Payment Service aborts transaction. "
                    "Fraud Model v3 deployed at 13:45 — increased CPU by 40%. "
                    "Under high traffic, response time exceeds 2 second threshold."
                ),
                "tags": ["payment", "service", "fraud", "critical", "ml"],
                "children": [],
                "parents": ["payment_service"],
                "metadata": {
                    "port": 8090,
                    "owner": "risk-team",
                    "model_version": "v3",
                    "timeout_ms": 2000,
                },
            },
            # ── Inventory Service ─────────────────────────────────────────────
            {
                "node_id": "inventory_service",
                "node_type": "service",
                "title": "Inventory Service",
                "summary": "Tracks and manages product stock levels",
                "content": (
                    "Inventory Service validates stock availability during order creation. "
                    "Runs on port 8083. "
                    "Depends on Warehouse Service for physical stock data."
                ),
                "tags": ["inventory", "service", "orders"],
                "children": ["warehouse_service"],
                "parents": ["order_service"],
                "metadata": {"port": 8083, "owner": "inventory-team"},
            },
            # ── Warehouse Service ─────────────────────────────────────────────
            {
                "node_id": "warehouse_service",
                "node_type": "service",
                "title": "Warehouse Service",
                "summary": "Manages physical warehouse and fulfillment",
                "content": (
                    "Warehouse Service tracks physical stock and manages fulfillment. "
                    "Runs on port 8084. "
                    "Not on critical payment or order confirmation path."
                ),
                "tags": ["warehouse", "service", "fulfillment"],
                "children": [],
                "parents": ["inventory_service"],
                "metadata": {"port": 8084, "owner": "logistics-team"},
            },
            # ── Notification Service ──────────────────────────────────────────
            {
                "node_id": "notification_service",
                "node_type": "service",
                "title": "Notification Service",
                "summary": "Sends order confirmations and alerts",
                "content": (
                    "Notification Service sends order confirmations to customers. "
                    "Order Service waits up to 5 seconds for confirmation. "
                    "If Notification Service is unavailable, orders remain PENDING. "
                    "Depends on Email Service for delivery."
                ),
                "tags": ["notification", "service", "orders", "critical"],
                "children": ["email_service"],
                "parents": ["order_service"],
                "metadata": {"port": 8085, "owner": "comms-team", "timeout_ms": 5000},
            },
            # ── Email Service ─────────────────────────────────────────────────
            {
                "node_id": "email_service",
                "node_type": "service",
                "title": "Email Service",
                "summary": "Delivers transactional emails to customers",
                "content": (
                    "Email Service delivers order confirmations, receipts, and alerts. "
                    "Runs on port 8086. "
                    "Email Service v1.2 deployed at 12:00 — contained template rendering bug. "
                    "Service crashed at 12:15 due to unhandled exception in template engine."
                ),
                "tags": ["email", "service", "notification", "orders", "critical"],
                "children": [],
                "parents": ["notification_service"],
                "metadata": {
                    "port": 8086,
                    "owner": "comms-team",
                    "version": "1.2",
                    "crashed_at": "12:15",
                },
            },
        ]

    @staticmethod
    def _database_nodes() -> list[dict]:
        return [
            # ── Payment DB ────────────────────────────────────────────────────
            {
                "node_id": "payment_db",
                "node_type": "database",
                "title": "Payment Database",
                "summary": "Stores all payment transaction records",
                "content": (
                    "PostgreSQL database on port 5432. "
                    "Max connection pool: 100 connections. "
                    "Fraud Model v3 increased CPU usage on Payment Service by 40%, "
                    "causing connection pool exhaustion under high load. "
                    "When pool is exhausted, Payment Service returns 503."
                ),
                "tags": ["payment", "database", "critical"],
                "children": [],
                "parents": ["payment_service"],
                "metadata": {
                    "port": 5432,
                    "type": "postgresql",
                    "owner": "payments-team",
                    "max_connections": 100,
                },
            },
            # ── Orders DB ─────────────────────────────────────────────────────
            {
                "node_id": "orders_db",
                "node_type": "database",
                "title": "Orders Database",
                "summary": "Stores all order records and status",
                "content": (
                    "PostgreSQL database on port 5433. "
                    "Max connection pool: 50 connections. "
                    "Stores order status — PENDING, CONFIRMED, SHIPPED, DELIVERED. "
                    "Orders remain PENDING when Notification Service is unavailable."
                ),
                "tags": ["orders", "database", "critical"],
                "children": [],
                "parents": ["order_service"],
                "metadata": {
                    "port": 5433,
                    "type": "postgresql",
                    "owner": "orders-team",
                    "max_connections": 50,
                },
            },
            # ── User DB ───────────────────────────────────────────────────────
            {
                "node_id": "user_db",
                "node_type": "database",
                "title": "User Database",
                "summary": "Stores user accounts and authentication data",
                "content": (
                    "MongoDB database on port 27017. "
                    "Stores user profiles, credentials, and session data. "
                    "Not on the critical payment or order path."
                ),
                "tags": ["users", "database", "authentication"],
                "children": [],
                "parents": ["user_service"],
                "metadata": {
                    "port": 27017,
                    "type": "mongodb",
                    "owner": "identity-team",
                },
            },
        ]

    @staticmethod
    def _deployment_nodes() -> list[dict]:
        return [
            # ── Flash Sale Deployment ─────────────────────────────────────────
            {
                "node_id": "flash_sale_deployment",
                "node_type": "deployment",
                "title": "Flash Sale v2.1 Deployment",
                "summary": "Increased traffic limits 10x at 14:00",
                "content": (
                    "Flash Sale v2.1 deployed at 14:00. "
                    "Increased API Gateway traffic limits from 1000 to 10000 req/sec. "
                    "This caused a 10x traffic spike to Payment Service. "
                    "Payment Service was not scaled to handle this load. "
                    "Fraud Detection response time exceeded 2 second threshold at 14:03. "
                    "Payment Service began returning 503 errors at 14:05."
                ),
                "tags": ["deployment", "payment", "critical", "traffic"],
                "children": [],
                "parents": ["api_gateway"],
                "metadata": {
                    "version": "2.1",
                    "deployed_at": "14:00",
                    "deployed_by": "platform-team",
                    "traffic_multiplier": 10,
                },
            },
            # ── Fraud Model Deployment ────────────────────────────────────────
            {
                "node_id": "fraud_model_deployment",
                "node_type": "deployment",
                "title": "Fraud Model v3 Deployment",
                "summary": "Updated fraud scoring model deployed at 13:45",
                "content": (
                    "Fraud Model v3 deployed at 13:45. "
                    "New model increased CPU usage on Fraud Detection Service by 40%. "
                    "Under normal load this is acceptable. "
                    "Under Flash Sale 10x traffic, CPU spike caused "
                    "Payment DB connection pool exhaustion. "
                    "First Payment DB errors observed at 14:06."
                ),
                "tags": ["deployment", "payment", "fraud", "ml", "database"],
                "children": [],
                "parents": ["fraud_detection"],
                "metadata": {
                    "version": "v3",
                    "deployed_at": "13:45",
                    "deployed_by": "risk-team",
                    "cpu_increase_pct": 40,
                },
            },
            # ── Email Service Deployment ──────────────────────────────────────
            {
                "node_id": "email_deployment",
                "node_type": "deployment",
                "title": "Email Service v1.2 Deployment",
                "summary": "Bug fix deployment that introduced template rendering crash",
                "content": (
                    "Email Service v1.2 deployed at 12:00. "
                    "Intended as a minor bug fix for email formatting. "
                    "Introduced unhandled exception in template rendering engine. "
                    "Service crashed at 12:15 when first order confirmation was sent. "
                    "Notification Service began timing out at 12:16. "
                    "Order Service orders stuck in PENDING state from 12:16 onwards."
                ),
                "tags": ["deployment", "email", "notification", "orders"],
                "children": [],
                "parents": ["email_service"],
                "metadata": {
                    "version": "1.2",
                    "deployed_at": "12:00",
                    "deployed_by": "comms-team",
                    "crashed_at": "12:15",
                },
            },
        ]
