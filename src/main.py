from datetime import datetime
from contextlib import asynccontextmanager

import msgspec
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from src.search import VectorSearch
from src.vectorizer import Vectorizer


class TransactionPayload(msgspec.Struct):
    amount: float
    installments: int
    requested_at: datetime


class CustomerPayload(msgspec.Struct):
    avg_amount: float
    tx_count_24h: int
    known_merchants: list[str]


class MerchantPayload(msgspec.Struct):
    id: str
    mcc: str
    avg_amount: float


class TerminalPayload(msgspec.Struct):
    is_online: bool
    card_present: bool
    km_from_home: float


class LastTransactionPayload(msgspec.Struct):
    timestamp: datetime
    km_from_current: float


class FraudScoreRequest(msgspec.Struct):
    id: str
    transaction: TransactionPayload
    customer: CustomerPayload
    merchant: MerchantPayload
    terminal: TerminalPayload
    last_transaction: LastTransactionPayload | None = None


decoder = msgspec.json.Decoder(type=FraudScoreRequest)


@asynccontextmanager
async def lifespan(app: Starlette):
    app.state.ready = False
    search = VectorSearch()
    search.warmup()
    app.state.search = search
    app.state.vectorizer = Vectorizer()
    app.state.ready = True
    yield


def ready(request: Request) -> Response:
    if not getattr(request.app.state, "ready", False):
        return Response(status_code=503)
    return Response(status_code=200)


async def fraud_score(request: Request) -> Response:
    if not getattr(request.app.state, "ready", False):
        return Response(status_code=503)

    try:
        payload = decoder.decode(await request.body())
    except msgspec.DecodeError:
        return Response(status_code=400)

    vectorizer: Vectorizer = request.app.state.vectorizer
    search: VectorSearch = request.app.state.search

    vector = vectorizer.transaction_to_vector(payload)
    fraud_score, approved = search.score(vector, k=5)

    body = msgspec.json.encode(
        {
            "approved": approved,
            "fraud_score": round(fraud_score, 4),
        }
    )
    return Response(body, media_type="application/json")


app = Starlette(
    lifespan=lifespan,
    routes=[
        Route("/ready", ready, methods=["GET"]),
        Route("/fraud-score", fraud_score, methods=["POST"]),
    ]
)
