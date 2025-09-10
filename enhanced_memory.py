from mem0 import Memory
import json
from typing import Dict, List, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class EnhancedProgrammingMemory:
    """Advanced memory management for programming queries"""

    def __init__(self, config: Dict[str, Any]):
        try:
            self.memory = Memory.from_config(config)
            self.session_context = {}
            self.learning_patterns = {}
            logger.info("Enhanced memory system initialized")
        except Exception as e:
            logger.error(f"Failed to initialize enhanced memory: {e}")
            self.memory = None

    def store_programming_context(self, query: str, result, user_id: str):
        """Store rich contextual information about programming queries"""
        if not self.memory:
            return

        try:
            # Extract programming concepts and relationships
            context_data = {
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'detected_languages': getattr(result, 'detected_languages', []),
                'concepts': getattr(result, 'concepts', []),
                'code_snippets_count': len(getattr(result, 'code_snippets', [])),
                'confidence_score': getattr(result, 'confidence_score', 0.0),
                'search_strategies': getattr(result, 'query_strategies_used', [])
            }

            # Store conceptual relationships
            for concept in context_data['concepts']:
                concept_memory = f"User frequently asks about {concept} in context of {query}"
                self.memory.add(
                    concept_memory,
                    user_id=user_id,
                    metadata={'type': 'concept', 'concept': concept}
                )

            # Store language preferences
            for lang in context_data['detected_languages']:
                lang_preference = f"User works with {lang} programming language"
                self.memory.add(
                    lang_preference,
                    user_id=user_id,
                    metadata={'type': 'language_preference', 'language': lang}
                )

            # Store successful query patterns
            if context_data['confidence_score'] > 0.8:
                pattern_memory = f"Successful query pattern: {query} yielded high confidence results"
                self.memory.add(
                    pattern_memory,
                    user_id=user_id,
                    metadata={'type': 'successful_pattern', 'confidence': context_data['confidence_score']}
                )

            # Store learning progress
            self._update_learning_patterns(user_id, context_data)

        except Exception as e:
            logger.error(f"Failed to store programming context: {e}")

    def get_personalized_context(self, current_query: str, user_id: str) -> Dict[str, Any]:
        """Retrieve personalized context for current query"""
        if not self.memory:
            return {'context_strength': 0, 'related_concepts': [], 'language_preferences': [],
                    'successful_patterns': []}

        try:
            # Search for related concepts
            concept_memories = self.memory.search(
                current_query,
                user_id=user_id,
                limit=10
            )

            # Filter by memory types
            related_concepts = []
            language_preferences = []
            successful_patterns = []

            for mem in concept_memories:
                mem_text = mem.get('text', '')
                if 'frequently asks about' in mem_text:
                    related_concepts.append(mem_text)
                elif 'works with' in mem_text and 'programming language' in mem_text:
                    language_preferences.append(mem_text)
                elif 'Successful query pattern' in mem_text:
                    successful_patterns.append(mem_text)

            return {
                'related_concepts': related_concepts,
                'language_preferences': language_preferences,
                'successful_patterns': successful_patterns,
                'context_strength': len(concept_memories)
            }

        except Exception as e:
            logger.error(f"Failed to get personalized context: {e}")
            return {'context_strength': 0, 'related_concepts': [], 'language_preferences': [],
                    'successful_patterns': []}

    def get_personalization_strength(self, query: str, user_id: str) -> float:
        """Calculate how much personalization data is available"""
        try:
            context = self.get_personalized_context(query, user_id)
            strength = min(1.0, context['context_strength'] / 10.0)  # Normalize to 0-1
            return strength
        except:
            return 0.0

    def _update_learning_patterns(self, user_id: str, context_data: Dict[str, Any]):
        """Update learning patterns based on query history"""
        if user_id not in self.learning_patterns:
            self.learning_patterns[user_id] = {
                'query_count': 0,
                'favorite_languages': {},
                'frequent_concepts': {},
                'learning_trajectory': []
            }

        patterns = self.learning_patterns[user_id]
        patterns['query_count'] += 1

        # Update language frequency
        for lang in context_data['detected_languages']:
            patterns['favorite_languages'][lang] = patterns['favorite_languages'].get(lang, 0) + 1

        # Update concept frequency
        for concept in context_data['concepts']:
            patterns['frequent_concepts'][concept] = patterns['frequent_concepts'].get(concept, 0) + 1

        # Track learning trajectory
        patterns['learning_trajectory'].append({
            'timestamp': context_data['timestamp'],
            'query': context_data['query'][:100],  # Truncate for privacy
            'confidence': context_data['confidence_score']
        })

        # Keep only recent trajectory (last 50 queries)
        patterns['learning_trajectory'] = patterns['learning_trajectory'][-50:]

    def get_learning_insights(self, user_id: str) -> Dict[str, Any]:
        """Get insights about user's learning patterns"""
        if user_id not in self.learning_patterns:
            return {}

        patterns = self.learning_patterns[user_id]

        # Find top languages
        top_languages = sorted(
            patterns['favorite_languages'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        # Find top concepts
        top_concepts = sorted(
            patterns['frequent_concepts'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        # Calculate learning velocity (improvement over time)
        trajectory = patterns['learning_trajectory']
        if len(trajectory) >= 10:
            recent_confidence = sum([t['confidence'] for t in trajectory[-10:]]) / 10
            older_confidence = sum([t['confidence'] for t in trajectory[-20:-10]]) / 10 if len(
                trajectory) >= 20 else recent_confidence
            learning_velocity = recent_confidence - older_confidence
        else:
            learning_velocity = 0.0

        return {
            'total_queries': patterns['query_count'],
            'top_languages': top_languages,
            'top_concepts': top_concepts,
            'learning_velocity': learning_velocity,
            'expertise_areas': [lang for lang in top_languages[:3]]
        }

    def suggest_learning_topics(self, user_id: str, current_query: str) -> List[str]:
        """Suggest related topics based on learning patterns"""
        try:
            insights = self.get_learning_insights(user_id)
            suggestions = []

            # Suggest advanced topics in favorite languages
            for lang, _ in insights.get('top_languages', [])[:2]:
                suggestions.extend([
                    f"Advanced {lang} patterns",
                    f"{lang} performance optimization",
                    f"{lang} testing strategies"
                ])

            # Suggest related concepts
            query_lower = current_query.lower()
            if 'basic' in query_lower or 'beginner' in query_lower:
                suggestions.extend([
                    "Intermediate programming concepts",
                    "Code organization best practices",
                    "Debugging techniques"
                ])

            return suggestions[:5]  # Limit suggestions

        except Exception as e:
            logger.error(f"Failed to generate suggestions: {e}")
            return []

    def clear_user_memory(self, user_id: str):
        """Clear memory for a specific user"""
        try:
            if self.memory:
                # This would need to be implemented based on mem0's API
                # For now, just clear local patterns
                if user_id in self.learning_patterns:
                    del self.learning_patterns[user_id]
                logger.info(f"Cleared memory for user: {user_id}")
        except Exception as e:
            logger.error(f"Failed to clear user memory: {e}")
