class ApplicationError(Exception):
    code = "APPLICATION_ERROR"


class ConversationNotFound(ApplicationError):
    code = "CONVERSATION_NOT_FOUND"


class CitationNotFound(ApplicationError):
    code = "CITATION_NOT_FOUND"


class InvalidInput(ApplicationError):
    code = "INVALID_INPUT"


class PayloadTooLarge(InvalidInput):
    code = "PAYLOAD_TOO_LARGE"


class ServiceUnavailable(ApplicationError):
    """Something the request needs is not ready or not configured (HTTP 503)."""

    code = "SERVICE_UNAVAILABLE"


class ServiceStarting(ServiceUnavailable):
    code = "SERVICE_STARTING"


class ServiceNotConfigured(ServiceUnavailable):
    code = "SERVICE_NOT_CONFIGURED"


class AgentUnavailable(ServiceUnavailable):
    code = "AGENT_UNAVAILABLE"
