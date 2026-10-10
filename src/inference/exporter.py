"""Prediction export for the inference subsystem (PRD 91).

Writes predictions to CSV and JSON, plus a run summary. Exports are pure
serialisation: no numerical computation beyond formatting is performed here.
"""

import csv
import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence

from src.inference.exceptions import PredictionExportError
from src.inference.predictor import BatchPredictionResult, PredictionResult

logger = logging.getLogger(__name__)

CSV_COLUMNS = (
    "sample_id",
    "formula",
    "formation_energy_ev_per_atom",
    "num_atoms",
    "num_edges",
    "source_path",
)


def _ensure_parent(path: str) -> None:
    """Creates the parent directory of a target path when needed."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


class PredictionExporter:
    """Serialises predictions to CSV and JSON artifacts."""

    @staticmethod
    def to_csv(predictions: Sequence[PredictionResult], path: str) -> str:
        """Writes predictions to a CSV file.

        Args:
            predictions: Predictions to export.
            path: Destination ``.csv`` path.

        Returns:
            str: The path written.

        Raises:
            PredictionExportError: If the file cannot be written.
        """
        try:
            _ensure_parent(path)
            with open(path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
                writer.writeheader()
                for prediction in predictions:
                    row = prediction.to_dict()
                    writer.writerow({key: row.get(key) for key in CSV_COLUMNS})
        except OSError as exc:
            raise PredictionExportError(path, str(exc)) from exc

        logger.info(f"Wrote {len(predictions)} predictions to '{path}'.")
        return path

    @staticmethod
    def to_json(
        predictions: Sequence[PredictionResult],
        path: str,
        summary: Optional[Dict[str, Any]] = None,
        failures: Optional[Sequence[Any]] = None,
    ) -> str:
        """Writes predictions to a JSON file.

        Args:
            predictions: Predictions to export.
            path: Destination ``.json`` path.
            summary: Optional aggregate statistics for the run.
            failures: Optional ``(sample_id, reason)`` pairs for skipped
                structures.

        Returns:
            str: The path written.

        Raises:
            PredictionExportError: If the file cannot be written.
        """
        payload: Dict[str, Any] = {
            "predictions": [prediction.to_dict() for prediction in predictions],
        }
        if summary is not None:
            payload["summary"] = summary
        if failures is not None:
            payload["failures"] = [
                {"sample_id": sample_id, "reason": reason}
                for sample_id, reason in failures
            ]

        try:
            _ensure_parent(path)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except OSError as exc:
            raise PredictionExportError(path, str(exc)) from exc

        logger.info(f"Wrote {len(predictions)} predictions to '{path}'.")
        return path

    @classmethod
    def export(
        cls,
        result: BatchPredictionResult,
        output_dir: str,
        basename: str = "predictions",
    ) -> List[str]:
        """Writes the CSV and JSON artifacts for a batch run.

        Args:
            result: The batch prediction outcome to export.
            output_dir: Directory the artifacts are written into.
            basename: Stem for the generated files (default: 'predictions').

        Returns:
            List[str]: Paths of the written artifacts, CSV first.

        Raises:
            PredictionExportError: If any artifact cannot be written.
        """
        csv_path = os.path.join(output_dir, f"{basename}.csv")
        json_path = os.path.join(output_dir, f"{basename}.json")

        cls.to_csv(result.predictions, csv_path)
        cls.to_json(
            result.predictions,
            json_path,
            summary=result.summary(),
            failures=result.failures,
        )
        return [csv_path, json_path]

    @classmethod
    def export_single(
        cls,
        prediction: PredictionResult,
        output_path: str,
    ) -> str:
        """Writes one prediction to a single CSV or JSON file.

        Args:
            prediction: The prediction to export.
            output_path: Destination path; the extension selects the format.

        Returns:
            str: The path written.

        Raises:
            PredictionExportError: If the extension is unsupported or the file
                cannot be written.
        """
        extension = os.path.splitext(output_path)[1].lower()
        if extension == ".csv":
            return cls.to_csv([prediction], output_path)
        if extension == ".json":
            return cls.to_json([prediction], output_path)
        raise PredictionExportError(
            output_path, f"unsupported export format '{extension}' (use .csv or .json)"
        )
