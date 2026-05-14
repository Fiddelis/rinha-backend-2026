from datetime import UTC, datetime
from typing import Any

import numpy as np

NORMALIZATION = {
    "max_amount": 10000.0,
    "max_installments": 12.0,
    "amount_vs_avg_ratio": 10.0,
    "max_minutes": 1440.0,
    "max_km": 1000.0,
    "max_tx_count_24h": 20.0,
    "max_merchant_avg_amount": 10000.0,
}

MCC_RISK = {
    "5411": 0.15,
    "5812": 0.30,
    "5912": 0.20,
    "5944": 0.45,
    "7801": 0.80,
    "7802": 0.75,
    "7995": 0.85,
    "4511": 0.35,
    "5311": 0.25,
    "5999": 0.50,
}


class Vectorizer:
    def __init__(self) -> None:
        self.normalization = NORMALIZATION
        self.mcc_risk = MCC_RISK

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))

    @staticmethod
    def _as_utc(dt: str | datetime) -> datetime:
        if isinstance(dt, str):
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)

    @staticmethod
    def _field(obj: Any, name: str) -> Any:
        if isinstance(obj, dict):
            return obj[name]
        return getattr(obj, name)

    def transaction_to_vector(self, payload: Any) -> np.ndarray:
        transaction = self._field(payload, "transaction")
        customer = self._field(payload, "customer")
        merchant = self._field(payload, "merchant")
        terminal = self._field(payload, "terminal")
        last_transaction = (
            payload.get("last_transaction")
            if isinstance(payload, dict)
            else getattr(payload, "last_transaction")
        )

        requested_at = self._as_utc(self._field(transaction, "requested_at"))

        customer_avg_amount = self._field(customer, "avg_amount")
        transaction_amount = self._field(transaction, "amount")
        amount_vs_avg = 0.0
        if customer_avg_amount > 0:
            amount_vs_avg = self._clamp(
                (transaction_amount / customer_avg_amount)
                / self.normalization["amount_vs_avg_ratio"]
            )

        minutes_since_last_tx = -1.0
        km_from_last_tx = -1.0
        if last_transaction is not None:
            last_timestamp = self._as_utc(self._field(last_transaction, "timestamp"))
            delta_minutes = max(
                0.0, (requested_at - last_timestamp).total_seconds() / 60
            )
            minutes_since_last_tx = self._clamp(
                delta_minutes / self.normalization["max_minutes"]
            )
            km_from_last_tx = self._clamp(
                self._field(last_transaction, "km_from_current")
                / self.normalization["max_km"]
            )

        known_merchants = self._field(customer, "known_merchants")
        merchant_id = self._field(merchant, "id")

        vector = np.array(
            [
                self._clamp(transaction_amount / self.normalization["max_amount"]),
                self._clamp(
                    self._field(transaction, "installments")
                    / self.normalization["max_installments"]
                ),
                amount_vs_avg,
                requested_at.hour / 23,
                requested_at.weekday() / 6,
                minutes_since_last_tx,
                km_from_last_tx,
                self._clamp(
                    self._field(terminal, "km_from_home")
                    / self.normalization["max_km"]
                ),
                self._clamp(
                    self._field(customer, "tx_count_24h")
                    / self.normalization["max_tx_count_24h"]
                ),
                1.0 if self._field(terminal, "is_online") else 0.0,
                1.0 if self._field(terminal, "card_present") else 0.0,
                0.0 if merchant_id in known_merchants else 1.0,
                float(self.mcc_risk.get(self._field(merchant, "mcc"), 0.5)),
                self._clamp(
                    self._field(merchant, "avg_amount")
                    / self.normalization["max_merchant_avg_amount"]
                ),
            ],
            dtype=np.float32,
        )

        return vector
