class DomainError(Exception):
    """Base class for every error raised by the domain layer."""

    code: str = "DOMAIN_ERROR"

    def __init__(self, message: str = "", *, code: str | None = None) -> None:
        super().__init__(message or self.__class__.__name__)
        if code is not None:
            self.code = code
