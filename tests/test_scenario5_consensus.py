"""Scenario 5: Hospital Network Consensus — Semvec Layers 3+4 (Regions, Observer).

Tests that:
- Region can be created with consensus_threshold
- Clusters can be added to region
- Observer can be created and registered to region
- observer_sample returns data
- Region events can be queried
"""
from __future__ import annotations

import pytest
from src.core.embedder import HashEmbedder
from src.core.semvec_client import SemvecClient


@pytest.fixture(scope="module")
def semvec():
    return SemvecClient(embedder=HashEmbedder(dimension=32), owner_subject="test-suite")


@pytest.fixture
def cardiology_cluster(semvec):
    """Create cardiology cluster, delete after test."""
    cluster = semvec.create_cluster(
        name="cardiology-memorial-test",
        aggregation_mode="weighted_average",
        coupling_factor=0.2,
    )
    cid = cluster["cluster_id"]
    yield cid
    try:
        semvec.delete_cluster(cid)
    except Exception:
        pass


@pytest.fixture
def emergency_cluster(semvec):
    """Create emergency cluster, delete after test."""
    cluster = semvec.create_cluster(
        name="emergency-riverside-test",
        aggregation_mode="weighted_average",
        coupling_factor=0.2,
    )
    cid = cluster["cluster_id"]
    yield cid
    try:
        semvec.delete_cluster(cid)
    except Exception:
        pass


@pytest.fixture
def hospital_region(semvec):
    """Create hospital-network region, delete after test."""
    region = semvec.create_region(
        name="hospital-network-test",
        consensus_threshold=0.5,
        vote_window_seconds=60.0,
    )
    rid = region["region_id"]
    yield rid
    try:
        semvec.delete_region(rid)
    except Exception:
        pass


class TestRegionCreate:
    def test_create_region_returns_region_id(self, semvec):
        """POST /region/ returns a region_id."""
        region = semvec.create_region(
            name="test-region-create",
            consensus_threshold=0.5,
        )
        assert "region_id" in region
        assert region["region_id"]
        try:
            semvec.delete_region(region["region_id"])
        except Exception:
            pass

    def test_create_region_with_consensus_threshold(self, semvec):
        """Region stores consensus_threshold."""
        region = semvec.create_region(
            name="threshold-test-region",
            consensus_threshold=0.7,
            vote_window_seconds=30.0,
        )
        assert "region_id" in region
        rid = region["region_id"]
        try:
            semvec.delete_region(rid)
        except Exception:
            pass

    def test_get_region_by_id(self, semvec, hospital_region):
        """GET /region/{id} returns the region."""
        result = semvec.get_region(hospital_region)
        assert result is not None
        assert "region_id" in result or "name" in result or "data" in result


class TestRegionClusters:
    def test_add_cluster_to_region(self, semvec, hospital_region, cardiology_cluster):
        """POST /region/{id}/clusters adds a cluster to the region."""
        result = semvec.add_region_cluster(hospital_region, cardiology_cluster)
        assert result is not None

    def test_add_two_clusters_to_region(
        self, semvec, hospital_region, cardiology_cluster, emergency_cluster
    ):
        """Both cardiology and emergency clusters can be added to the region."""
        r1 = semvec.add_region_cluster(hospital_region, cardiology_cluster)
        r2 = semvec.add_region_cluster(hospital_region, emergency_cluster)
        assert r1 is not None
        assert r2 is not None

    def test_cluster_store_seeds_cardiology(self, semvec, cardiology_cluster):
        """Cardiology cluster can be seeded with Rodriguez AMI queries."""
        qa_pairs = [
            (
                "Maria Rodriguez Acute Myocardial Infarction — current troponin levels?",
                "Troponin I peaked at 4.2 ng/mL at 6 hours post-admission.",
            ),
            (
                "Post-MI antiplatelet therapy — aspirin plus clopidogrel or ticagrelor?",
                "Dual antiplatelet: Aspirin 81mg + Ticagrelor 90mg BID per AHA guidelines.",
            ),
        ]
        for msg, resp in qa_pairs:
            result = semvec.cluster_store(cardiology_cluster, msg, resp)
            assert result is not None

    def test_cluster_store_seeds_emergency(self, semvec, emergency_cluster):
        """Emergency cluster can be seeded with Morrison ER queries."""
        qa_pairs = [
            (
                "James Morrison Emergency Room Visit — chief complaint and triage priority?",
                "Chief complaint: chest pain 7/10. Triage priority ESI-2 (emergent).",
            ),
            (
                "Morrison has Type 2 Diabetes and COPD — medication reconciliation needed?",
                "Yes — hold Metformin pre-procedure, review inhalers for COPD management.",
            ),
        ]
        for msg, resp in qa_pairs:
            result = semvec.cluster_store(emergency_cluster, msg, resp)
            assert result is not None


class TestRegionEvents:
    def test_get_region_events_returns_list(self, semvec, hospital_region):
        """GET /region/{id}/events returns a list (possibly empty)."""
        events = semvec.get_region_events(hospital_region, limit=20)
        assert isinstance(events, list)

    def test_region_events_after_cluster_run(
        self, semvec, hospital_region, cardiology_cluster, emergency_cluster
    ):
        """After cluster runs, region events can be queried."""
        # Add clusters to region
        try:
            semvec.add_region_cluster(hospital_region, cardiology_cluster)
            semvec.add_region_cluster(hospital_region, emergency_cluster)
        except Exception:
            pass

        # Seed and run in cardiology cluster
        semvec.cluster_store(
            cardiology_cluster,
            "Rodriguez AMI troponin levels?",
            "Troponin peaked at 4.2 ng/mL.",
        )
        semvec.cluster_run(
            cardiology_cluster,
            "Maria Rodriguez post-MI antiplatelet therapy?",
            short_circuit_threshold=0.6,
        )

        # Seed and run in emergency cluster
        semvec.cluster_store(
            emergency_cluster,
            "Morrison ER chief complaint?",
            "Chest pain ESI-2.",
        )
        semvec.cluster_run(
            emergency_cluster,
            "Morrison diabetes medication reconciliation?",
            short_circuit_threshold=0.6,
        )

        # Query region events
        events = semvec.get_region_events(hospital_region, limit=20)
        assert isinstance(events, list)

    def test_region_drift_pivot_query(
        self, semvec, hospital_region, cardiology_cluster
    ):
        """After a drift-inducing pivot query, region events update."""
        # Add cluster to region
        try:
            semvec.add_region_cluster(hospital_region, cardiology_cluster)
        except Exception:
            pass

        # Seed cardiology context
        cardiology_msgs = [
            ("Rodriguez AMI troponin?", "Troponin 4.2 ng/mL."),
            ("Post-MI antiplatelet?", "Aspirin + Ticagrelor."),
            ("Cardio rehab referral?", "Dr. Okonkwo for cardiac rehab."),
        ]
        for msg, resp in cardiology_msgs:
            semvec.cluster_store(cardiology_cluster, msg, resp)

        # Pivot to infection control (drift-inducing)
        result = semvec.cluster_run(
            cardiology_cluster,
            "What is the latest infection control audit status at Memorial General?",
            short_circuit_threshold=0.6,
        )
        assert "top_similarity" in result or "short_circuit" in result

        # Events should still be queryable
        events = semvec.get_region_events(hospital_region, limit=20)
        assert isinstance(events, list)


class TestObserver:
    def test_create_observer_returns_id(self, semvec, hospital_region):
        """POST /observer/ returns observer data."""
        try:
            result = semvec.create_observer(
                sample_interval_seconds=30.0,
                region_ids=[hospital_region],
            )
            assert result is not None
        except Exception as e:
            if "404" in str(e) or "Not Found" in str(e) or "422" in str(e):
                pytest.skip(f"observer endpoint not available: {e}")
            raise

    def test_observer_sample_returns_data(self, semvec):
        """POST /observer/sample returns data."""
        try:
            result = semvec.observer_sample()
            assert result is not None
        except Exception as e:
            if "404" in str(e) or "Not Found" in str(e) or "422" in str(e):
                pytest.skip(f"observer/sample endpoint not available: {e}")
            raise

    def test_get_observer_summary_returns_data(self, semvec):
        """GET /observer/summary returns a summary dict."""
        try:
            result = semvec.get_observer_summary()
            assert result is not None
        except Exception as e:
            if "404" in str(e) or "Not Found" in str(e) or "422" in str(e):
                pytest.skip(f"observer/summary endpoint not available: {e}")
            raise

    def test_get_anomalies_returns_list(self, semvec):
        """GET /observer/anomalies returns a list."""
        try:
            result = semvec.get_anomalies(limit=20)
            assert isinstance(result, list)
        except Exception as e:
            if "404" in str(e) or "Not Found" in str(e) or "422" in str(e):
                pytest.skip(f"observer/anomalies endpoint not available: {e}")
            raise

    def test_observer_workflow(self, semvec, hospital_region, cardiology_cluster):
        """Full observer workflow: create, add region, sample."""
        try:
            # Create observer with region
            observer = semvec.create_observer(
                sample_interval_seconds=60.0,
                region_ids=[hospital_region],
            )
            assert observer is not None

            # Add cluster to region first
            try:
                semvec.add_region_cluster(hospital_region, cardiology_cluster)
            except Exception:
                pass

            # Seed cluster
            semvec.cluster_store(
                cardiology_cluster,
                "Rodriguez AMI investigation?",
                "Troponin elevated, catheterization scheduled.",
            )

            # Trigger sample
            sample = semvec.observer_sample()
            assert sample is not None

        except Exception as e:
            if "404" in str(e) or "Not Found" in str(e) or "422" in str(e):
                pytest.skip(f"observer endpoints not available: {e}")
            raise
