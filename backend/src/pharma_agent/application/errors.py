class ApplicationError(Exception):
    code = "APPLICATION_ERROR"


class ConversationNotFound(ApplicationError):
    code = "CONVERSATION_NOT_FOUND"


class InvalidInput(ApplicationError):
    code = "INVALID_INPUT"


class PayloadTooLarge(InvalidInput):
    code = "PAYLOAD_TOO_LARGE"


class AgentUnavailable(ApplicationError):
    code = "AGENT_UNAVAILABLE"
