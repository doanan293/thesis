import json

from sse_starlette import ServerSentEvent

from pharma_agent.application.progress import ProgressEvent


def to_server_sent_event(event: ProgressEvent) -> ServerSentEvent:
    return ServerSentEvent(
        data=json.dumps(event.data, ensure_ascii=False, default=str),
        event=event.type.value,
    )
