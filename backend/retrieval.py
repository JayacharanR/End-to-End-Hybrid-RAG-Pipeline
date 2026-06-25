"""Tri-Brid Retrieval Engine.

Implements the core retrieval logic including Qdrant Hybrid Search (Dense + Sparse)
and Reciprocal Rank Fusion (RRF) via the Qdrant Universal Query API.
"""

import logging
from typing import List, Dict, Any

from qdrant_client.http import models

from backend.config import get_settings
from backend.qdrant_client import get_async_qdrant
from data_pipeline.ingest import get_dense_model, get_sparse_model

logger = logging.getLogger(__name__)


async def hybrid_search(query: str) -> List[Dict[str, Any]]:
    """Execute a hybrid dense+sparse search using Qdrant Prefetch API for RRF.
    
    Args:
        query: The natural language search query.
        
    Returns:
        List of candidate documents with their RRF scores.
    """
    settings = get_settings()
    qdrant = get_async_qdrant()
    
    dense_model = get_dense_model()
    sparse_model = get_sparse_model()
    
    logger.debug("Generating dual embeddings for query: %s", query)
    
    # Generate embeddings
    dense_vector = list(dense_model.embed([query]))[0].tolist()
    sparse_obj = list(sparse_model.embed([query]))[0]
    sparse_vector = models.SparseVector(
        indices=sparse_obj.indices.tolist(),
        values=sparse_obj.values.tolist(),
    )
    
    # Qdrant Prefetch API for server-side Reciprocal Rank Fusion
    prefetch_dense = models.Prefetch(
        query=dense_vector,
        using="dense",
        limit=settings.rrf_k,
    )
    
    prefetch_sparse = models.Prefetch(
        query=sparse_vector,
        using="sparse",
        limit=settings.rrf_k,
    )
    
    logger.debug("Executing Qdrant Universal Query API with RRF fusion (k=%d)", settings.rrf_k)
    
    try:
        # The query method with multiple prefetches performs RRF by default
        results = await qdrant.query_points(
            collection_name=settings.qdrant_collection,
            prefetch=[prefetch_dense, prefetch_sparse],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=settings.retrieval_top_k,
            with_payload=True,
        )
        
        # Format results
        documents = []
        for point in results.points:
            documents.append({
                "id": point.id,
                "score": point.score,
                "title": point.payload.get("title", ""),
                "content": point.payload.get("page_content", ""),
                "url": point.payload.get("url", ""),
                "chunk_index": point.payload.get("chunk_index", 0),
            })
            
        logger.info("Hybrid search returned %d candidates.", len(documents))
        return documents
        
    except Exception as exc:
        logger.error("Hybrid search failed: %s", exc)
        return []
