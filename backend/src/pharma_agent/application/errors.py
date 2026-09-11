class ApplicationError(Exception):
    code = "APPLICATION_ERROR"


class ConversationNotFound(ApplicationError):
    code = "CONVERSATION_NOT_FOUND"


class InvalidInput(ApplicationError):
    code = "INVALID_INPUT"


class AgentUnavailable(ApplicationError):
    code = "AGENT_UNAVAILABLE"
