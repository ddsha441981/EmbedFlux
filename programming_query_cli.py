#!/usr/bin/env python3
"""
Enhanced CLI interface for programming notes query system.
Now includes web integration, advanced memory, and embedding model selection.
"""

import os
import sys
import argparse
import json
import time
import logging
import re
from typing import List, Dict, Any

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm

# Import our enhanced programming query engine
from programming_query_engine import (
    ProgrammingQueryEngine,
    ProgrammingQueryConfig,
    ProgrammingLanguage,
    PROGRAMMING_EMBEDDING_MODELS
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize rich console
console = Console()


class EnhancedProgrammingQueryCLI:
    """Enhanced CLI for programming query system with new features"""

    def __init__(self):
        self.engine = ProgrammingQueryEngine()
        self.config = ProgrammingQueryConfig()
        self.session_id = f"user_session_{int(time.time())}"

    def print_welcome(self):
        """Enhanced welcome message"""
        welcome_text = """
# 🚀 Enhanced Programming Notes Query System

Welcome to your **intelligent programming knowledge base**!

## ✨ New Features:
- 🧠 **Advanced Memory**: Learns your preferences and patterns
- 🌐 **Web Integration**: Includes results from Stack Overflow and documentation
- ⚡ **Smart Embeddings**: Choose from optimized models for better results
- 🎯 **Personalization**: Adapts to your programming language preferences

This system searches through your programming notes using multiple strategies and can now
augment results with web content and personalized recommendations.

**Supported Languages:** Python, JavaScript, Java, C++, C#, Go, Rust, PHP, Ruby, SQL, HTML, CSS, and more!
"""
        console.print(Panel(Markdown(welcome_text), border_style="blue", title="🎉 Enhanced System"))

    def print_help(self):
        """Enhanced help information"""
        help_text = """
## 📖 Available Commands

### Query Commands:
- **Any text** - Search your programming notes with AI enhancement
- **`help`** - Show this help message
- **`config`** - Configure search settings and features
- **`stats`** - Show detailed system statistics
- **`models`** - Compare and switch embedding models
- **`memory`** - View your learning insights and patterns
- **`clear`** - Clear screen
- **`cache`** - Manage query cache
- **`quit`** - Exit the program

### New Commands:
- **`web on/off`** - Enable/disable web search integration
- **`model <type>`** - Switch embedding models
- **`insights`** - Show personalized learning insights
- **`suggest`** - Get learning topic suggestions

### Example Queries:
- `"How to implement binary search in Python"`
- `"JavaScript async await examples"`
- `"SQL join types explanation"`
- `"React hooks best practices"`
- `"Java memory management"`

### Pro Tips:
- 🧠 System learns from your queries and provides personalized results
- 🌐 Enable web search for latest documentation and Stack Overflow answers
- ⚡ Try different embedding models for better results with `models` command
- 📊 Check your learning progress with `insights` command
"""
        console.print(Panel(Markdown(help_text), border_style="green", title="📚 Help Guide"))

    def print_enhanced_config(self):
        """Enhanced configuration display and modification"""
        console.print("\n⚙️ Enhanced Configuration:", style="bold yellow")

        config_table = Table(show_header=True, header_style="bold blue")
        config_table.add_column("Setting", style="dim", width=25)
        config_table.add_column("Value", style="bold", width=20)
        config_table.add_column("Description", style="dim", width=40)

        # Basic settings
        config_table.add_row("top_k_per_strategy", str(self.config.top_k_per_strategy), "Results per search strategy")
        config_table.add_row("final_top_k", str(self.config.final_top_k), "Final number of results")
        config_table.add_row("score_threshold", str(self.config.score_threshold), "Minimum similarity score")
        config_table.add_row("mmr_diversity", str(self.config.mmr_diversity), "MMR diversity factor")

        # Enhanced settings
        config_table.add_row("enable_web_search", str(self.config.enable_web_search), "Include web content in search")
        config_table.add_row("embedding_model_type", self.config.embedding_model_type, "Current embedding model")
        config_table.add_row("enhanced_memory", str(self.config.enhanced_memory), "Advanced memory features")
        config_table.add_row("max_web_results", str(self.config.max_web_results), "Maximum web results")

        console.print(config_table)

        # Enhanced configuration changes
        console.print("\n🔧 Change settings? (press Enter to skip)", style="bold cyan")

        try:
            # Basic settings
            new_top_k = Prompt.ask("Final top K", default=str(self.config.final_top_k))
            if new_top_k != str(self.config.final_top_k):
                self.config.final_top_k = int(new_top_k)

            # Web search toggle
            web_search = Confirm.ask("Enable web search integration?", default=self.config.enable_web_search)
            self.config.enable_web_search = web_search

            # Embedding model selection
            console.print("\n📊 Available embedding models:")
            for i, (key, model) in enumerate(PROGRAMMING_EMBEDDING_MODELS.items(), 1):
                status = "✅ Current" if key == self.config.embedding_model_type else ""
                console.print(f"  {i}. {key}: {model} {status}")

            model_choice = Prompt.ask(
                "Choose embedding model (1-5 or press Enter to keep current)",
                default="0"
            )

            if model_choice != "0" and model_choice.isdigit():
                model_keys = list(PROGRAMMING_EMBEDDING_MODELS.keys())
                if 1 <= int(model_choice) <= len(model_keys):
                    self.config.embedding_model_type = model_keys[int(model_choice) - 1]
                    console.print(f"✅ Switched to: {self.config.embedding_model_type}", style="green")

            console.print("✅ Configuration updated!", style="bold green")

        except (ValueError, KeyboardInterrupt) as e:
            if isinstance(e, ValueError):
                console.print(f"❌ Invalid input: {e}", style="bold red")
            else:
                console.print("\n❌ Configuration cancelled", style="dim")

    def print_enhanced_stats(self):
        """Enhanced system statistics"""
        stats = self.engine.get_stats()

        console.print("\n📊 Enhanced System Statistics:", style="bold yellow")

        # System status table
        system_table = Table(show_header=True, header_style="bold blue", title="System Status")
        system_table.add_column("Component", style="dim")
        system_table.add_column("Status", style="bold")
        system_table.add_column("Details", style="dim")

        system_table.add_row(
            "Chroma DB",
            "✅ Active" if stats['chroma_initialized'] else "❌ Inactive",
            f"Documents: {stats.get('chroma_count', 'unknown')}"
        )
        system_table.add_row(
            "Milvus DB",
            "✅ Active" if stats['milvus_initialized'] else "❌ Inactive",
            f"Documents: {stats.get('milvus_count', 'unknown')}"
        )
        system_table.add_row(
            "Groq LLM",
            "✅ Available" if stats['groq_llm_available'] else "❌ Unavailable",
            "For advanced synthesis"
        )
        system_table.add_row(
            "Enhanced Memory",
            "✅ Active" if stats['enhanced_memory_available'] else "❌ Inactive",
            "Personalization & learning"
        )
        system_table.add_row(
            "Web Processor",
            "✅ Ready" if stats['web_processor_available'] else "❌ Unavailable",
            "Jina Reader integration"
        )

        console.print(system_table)

        # Performance table
        perf_table = Table(show_header=True, header_style="bold green", title="Performance Metrics")
        perf_table.add_column("Metric", style="dim")
        perf_table.add_column("Value", style="bold")

        perf_table.add_row("Current Embedding Model", stats['current_embedding_type'])
        perf_table.add_row("Cache Size", str(stats['cache_size']))
        perf_table.add_row("Supported Languages", str(stats['supported_languages']))
        perf_table.add_row("Programming Concepts", str(stats['programming_concepts']))

        console.print(perf_table)

    def show_embedding_models(self):
        """Show and compare embedding models"""
        console.print("\n🤖 Embedding Models Comparison:", style="bold cyan")

        models_table = Table(show_header=True, header_style="bold blue")
        models_table.add_column("Model Type", style="dim")
        models_table.add_column("Model Name", style="bold")
        models_table.add_column("Status", style="green")
        models_table.add_column("Best For", style="dim")

        model_descriptions = {
            'fast-general': 'General queries, fast performance',
            'code-focused': 'Code understanding, syntax analysis',
            'multilingual-code': 'Multiple programming languages',
            'current-default': 'Balanced performance and accuracy',
            'programming-optimized': 'Best overall for programming (recommended)'
        }

        current_model = self.engine.current_embedding_type

        for model_type, model_name in PROGRAMMING_EMBEDDING_MODELS.items():
            status = "✅ Current" if model_type == current_model else "Available"
            models_table.add_row(
                model_type,
                model_name,
                status,
                model_descriptions.get(model_type, "Programming queries")
            )

        console.print(models_table)

        # Offer to test models
        if Confirm.ask("\n🧪 Would you like to compare models with a test query?"):
            test_query = Prompt.ask("Enter a test query", default="Python list comprehension")

            console.print(f"\n🔍 Testing models with query: '{test_query}'")
            with console.status("[bold green]Comparing models..."):
                comparison = self.engine.compare_embedding_models(test_query)

            # Display comparison results
            comparison_table = Table(show_header=True, header_style="bold yellow")
            comparison_table.add_column("Model", style="dim")
            comparison_table.add_column("Query Time (s)", style="bold")
            comparison_table.add_column("Results Found", style="green")
            comparison_table.add_column("Status", style="dim")

            for model, result in comparison.items():
                if 'error' in result:
                    comparison_table.add_row(model, "N/A", "Error", result['error'])
                else:
                    comparison_table.add_row(
                        model,
                        f"{result['query_time']:.3f}",
                        str(result['num_results']),
                        "✅ Success"
                    )

            console.print(comparison_table)

    def show_memory_insights(self):
        """Show learning insights and memory statistics"""
        if not self.engine.enhanced_memory:
            console.print("❌ Enhanced memory not available", style="bold red")
            return

        console.print("\n🧠 Your Learning Insights:", style="bold blue")

        try:
            insights = self.engine.enhanced_memory.get_learning_insights(self.engine.user_session_id)

            if not insights or insights.get('total_queries', 0) == 0:
                console.print("📚 No learning data yet. Keep asking questions!", style="dim")
                return

            # Learning overview
            overview_table = Table(show_header=True, header_style="bold green")
            overview_table.add_column("Metric", style="dim")
            overview_table.add_column("Value", style="bold")

            overview_table.add_row("Total Queries", str(insights.get('total_queries', 0)))
            overview_table.add_row("Learning Velocity", f"{insights.get('learning_velocity', 0):.2f}")
            overview_table.add_row("Expertise Areas", ", ".join(insights.get('expertise_areas', [])))

            console.print(overview_table)

            # Top languages
            if insights.get('top_languages'):
                console.print("\n🔤 Your Favorite Languages:", style="bold cyan")
                lang_table = Table(show_header=True, header_style="bold blue")
                lang_table.add_column("Language", style="bold")
                lang_table.add_column("Query Count", style="green")
                lang_table.add_column("Frequency", style="dim")

                total_queries = insights['total_queries']
                for lang, count in insights['top_languages'][:5]:
                    percentage = (count / total_queries) * 100
                    lang_table.add_row(lang, str(count), f"{percentage:.1f}%")

                console.print(lang_table)

            # Top concepts
            if insights.get('top_concepts'):
                console.print("\n🧠 Concepts You Explore:", style="bold cyan")
                concept_table = Table(show_header=True, header_style="bold blue")
                concept_table.add_column("Concept", style="bold")
                concept_table.add_column("Times Explored", style="green")

                for concept, count in insights['top_concepts'][:8]:
                    concept_table.add_row(concept, str(count))

                console.print(concept_table)

            # Learning suggestions
            suggestions = self.engine.enhanced_memory.suggest_learning_topics(
                self.engine.user_session_id,
                "general programming"
            )

            if suggestions:
                console.print("\n💡 Suggested Learning Topics:", style="bold yellow")
                for i, suggestion in enumerate(suggestions, 1):
                    console.print(f"  {i}. {suggestion}")

        except Exception as e:
            console.print(f"❌ Error retrieving insights: {e}", style="bold red")

    def display_enhanced_results(self, result, query: str):
        """Enhanced result display with new features"""
        if not result.documents:
            console.print("❌ No results found.", style="bold red")
            return

        # Enhanced header with new metrics
        memory_indicator = "🧠 Memory-Enhanced" if result.memory_context_used else "🤖 Standard"
        web_indicator = f"🌐 +{result.web_sources_count} web" if result.web_sources_count > 0 else ""

        header = f"\n🎯 **Answer from Your Programming Notes** ({memory_indicator}) {web_indicator}"
        console.print(header, style="bold green")

        # Display synthesized answer prominently
        if hasattr(result, 'synthesized_answer') and result.synthesized_answer:
            try:
                console.print(Panel(
                    Markdown(result.synthesized_answer),
                    title="📝 Complete Answer",
                    border_style="green",
                    padding=(1, 2)
                ))
            except Exception as e:
                console.print(Panel(
                    result.synthesized_answer,
                    title="📝 Complete Answer",
                    border_style="green"
                ))

            # Enhanced summary with new metrics
            summary_parts = [
                f"📊 **{len(result.documents)} sources**",
                f"⏱️ {result.query_time:.2f}s",
                f"🎯 Confidence: {result.confidence_score:.2f}",
                f"🧮 Model: {result.embedding_model_used.split('/')[-1]}"
            ]

            if result.personalization_strength > 0:
                summary_parts.append(f"👤 Personal: {result.personalization_strength:.2f}")

            console.print(" | ".join(summary_parts))

            if result.detected_languages:
                console.print(f"🔤 **Languages:** {', '.join(result.detected_languages)}")

            # Ask if user wants detailed view
            try:
                show_details = Confirm.ask("\n💡 Show detailed source breakdown?", default=False)
            except:
                show_details = False

            if show_details:
                self._display_detailed_sources(result, query)
        else:
            console.print("⚠️ Synthesized answer not available, showing source chunks:", style="bold yellow")
            self._display_detailed_sources(result, query)

    def _display_detailed_sources(self, result, query: str):
        """Display detailed source breakdown"""
        # Enhanced query details
        summary_text = f"""
**Query:** {query}
**Results:** {len(result.documents)} documents found in {result.query_time:.3f}s
**Strategies Used:** {', '.join(result.query_strategies_used)}
**Embedding Model:** {result.embedding_model_used}
"""
        console.print(Panel(Markdown(summary_text), title="📊 Query Details", border_style="blue"))

        # Categorize sources
        local_docs = []
        web_docs = []

        for i, (doc, score) in enumerate(zip(result.documents[:5], result.scores[:5])):
            if doc.metadata.get('content_type') == 'web':
                web_docs.append((doc, score, i + 1))
            else:
                local_docs.append((doc, score, i + 1))

        # Display local sources
        if local_docs:
            console.print("\n📚 **Local Sources:**", style="bold blue")
            for doc, score, idx in local_docs:
                content = doc.page_content
                if len(content) > 400:
                    content = content[:400] + "..."

                metadata_str = ""
                if doc.metadata:
                    metadata_items = [f"{k}: {v}" for k, v in doc.metadata.items()]
                    metadata_str = f"\n**Source:** {', '.join(metadata_items[:2])}"

                score_str = f"**Relevance:** {score:.3f}"
                result_text = f"{content}{metadata_str}\n{score_str}"

                console.print(Panel(
                    Markdown(result_text),
                    title=f"📄 Local Source {idx}",
                    border_style="dim"
                ))

        # Display web sources
        if web_docs:
            console.print("\n🌐 **Web Sources:**", style="bold cyan")
            for doc, score, idx in web_docs:
                content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                source_url = doc.metadata.get('source', 'Unknown')
                domain = doc.metadata.get('domain', 'Unknown')

                result_text = f"{content}\n**Source URL:** {source_url}\n**Domain:** {domain}\n**Relevance:** {score:.3f}"

                console.print(Panel(
                    Markdown(result_text),
                    title=f"🌐 Web Source {idx}",
                    border_style="cyan"
                ))

    def handle_quick_commands(self, query: str) -> bool:
        """Handle quick commands and return True if handled"""
        query_lower = query.lower().strip()

        if query_lower.startswith('web '):
            action = query_lower[4:].strip()
            if action in ['on', 'enable', 'true']:
                self.config.enable_web_search = True
                console.print("✅ Web search enabled!", style="bold green")
            elif action in ['off', 'disable', 'false']:
                self.config.enable_web_search = False
                console.print("✅ Web search disabled!", style="bold green")
            return True

        elif query_lower.startswith('model '):
            model_type = query_lower[6:].strip()
            if model_type in PROGRAMMING_EMBEDDING_MODELS:
                self.engine.switch_embedding_model(model_type)
                self.config.embedding_model_type = model_type
                console.print(f"✅ Switched to {model_type} model!", style="bold green")
            else:
                console.print(f"❌ Unknown model: {model_type}", style="bold red")
                self.show_embedding_models()
            return True

        elif query_lower in ['models', 'model']:
            self.show_embedding_models()
            return True

        elif query_lower in ['memory', 'insights']:
            self.show_memory_insights()
            return True

        elif query_lower in ['suggest', 'suggestions']:
            if self.engine.enhanced_memory:
                suggestions = self.engine.enhanced_memory.suggest_learning_topics(
                    self.engine.user_session_id,
                    "programming"
                )
                if suggestions:
                    console.print("\n💡 **Learning Suggestions:**", style="bold yellow")
                    for i, suggestion in enumerate(suggestions, 1):
                        console.print(f"  {i}. {suggestion}")
                else:
                    console.print("💡 Keep asking questions to get personalized suggestions!", style="dim")
            else:
                console.print("❌ Enhanced memory not available for suggestions", style="bold red")
            return True

        return False

    def manage_enhanced_cache(self):
        """Enhanced cache management"""
        console.print("\n🗄️ Enhanced Cache Management:", style="bold yellow")

        cache_stats = {
            'size': len(self.engine.query_cache),
            'models_available': len(PROGRAMMING_EMBEDDING_MODELS),
            'current_model': self.engine.current_embedding_type
        }

        stats_table = Table(show_header=True, header_style="bold blue")
        stats_table.add_column("Metric", style="dim")
        stats_table.add_column("Value", style="bold")

        stats_table.add_row("Cache Entries", str(cache_stats['size']))
        stats_table.add_row("Current Model", cache_stats['current_model'])
        stats_table.add_row("Available Models", str(cache_stats['models_available']))

        console.print(stats_table)

        if cache_stats['size'] > 0:
            action = Prompt.ask(
                "Choose action",
                choices=['clear', 'keep', 'info'],
                default='keep'
            )

            if action == 'clear':
                self.engine.clear_cache()
                console.print("✅ Cache cleared!", style="bold green")
            elif action == 'info':
                console.print(f"💾 Cache helps speed up repeated queries using the same model.", style="dim")
                console.print(f"💡 Cache is automatically cleared when switching embedding models.", style="dim")

    def run_interactive(self):
        """Enhanced interactive mode"""
        self.print_welcome()
        console.print("\n🎯 Ready for your enhanced programming questions!", style="bold green")
        console.print("Type 'help' for commands, 'config' for settings, or 'quit' to exit.\n")

        if self.engine.enhanced_memory:
            self.engine.user_session_id = self.session_id
            console.print("✨ Enhanced memory-powered session started!", style="bold green")

        if self.config.enable_web_search:
            console.print("🌐 Web search integration is active!", style="bold cyan")

        while True:
            try:
                # Get user input with enhanced prompt
                web_status = "🌐" if self.config.enable_web_search else ""
                memory_status = "🧠" if self.engine.enhanced_memory else ""
                model_status = f"⚡{self.config.embedding_model_type[:4]}"

                prompt_indicators = f"{web_status}{memory_status}{model_status}"
                query = console.input(f"\n[bold cyan]{prompt_indicators} ❓ Your question: [/bold cyan]").strip()

                if not query:
                    continue

                # Handle basic commands
                if query.lower() == 'quit':
                    console.print("👋 Goodbye! Happy coding!", style="bold blue")
                    break
                elif query.lower() == 'help':
                    self.print_help()
                    continue
                elif query.lower() == 'config':
                    self.print_enhanced_config()
                    continue
                elif query.lower() == 'stats':
                    self.print_enhanced_stats()
                    continue
                elif query.lower() == 'clear':
                    os.system('clear' if os.name == 'posix' else 'cls')
                    continue
                elif query.lower() == 'cache':
                    self.manage_enhanced_cache()
                    continue

                # Handle quick commands
                if self.handle_quick_commands(query):
                    continue

                # Execute enhanced query
                console.print("🔍 Searching your programming notes with AI enhancement...", style="dim")
                with console.status("[bold green]Querying enhanced databases..."):
                    result = self.engine.query(query, self.config)

                # Display enhanced results
                self.display_enhanced_results(result, query)

            except KeyboardInterrupt:
                console.print("\n\n👋 Goodbye! Happy coding!", style="bold blue")
                break
            except Exception as e:
                console.print(f"❌ Error: {e}", style="bold red")
                logger.error(f"CLI error: {e}", exc_info=True)

    # Keep existing methods for single query and batch processing...
    def run_single_query(self, query: str, output_format: str = "rich"):
        """Run a single enhanced query"""
        try:
            result = self.engine.query(query, self.config)

            if output_format == "json":
                # Enhanced JSON output
                output = {
                    "query": query,
                    "synthesized_answer": result.synthesized_answer,
                    "results": [{
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": score
                    } for doc, score in zip(result.documents, result.scores)],
                    "detected_languages": result.detected_languages,
                    "concepts": result.concepts,
                    "query_time": result.query_time,
                    "strategies_used": result.query_strategies_used,
                    "web_sources_count": result.web_sources_count,
                    "memory_context_used": result.memory_context_used,
                    "embedding_model": result.embedding_model_used,
                    "personalization_strength": result.personalization_strength
                }
                print(json.dumps(output, indent=2, ensure_ascii=False))
            else:
                self.display_enhanced_results(result, query)

        except Exception as e:
            console.print(f"❌ Error: {e}", style="bold red")
            return 1
        return 0


def main():
    """Enhanced main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="🚀 Enhanced Programming Notes Query System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Enhanced Examples:

# Interactive mode with all features
python programming_query_cli.py

# Single query with web search
python programming_query_cli.py -q "Python async patterns" --web

# Use specific embedding model
python programming_query_cli.py -q "Java collections" --model code-focused

# Batch processing with enhanced features
python programming_query_cli.py --batch queries.txt --output results.json --web

# JSON output with all enhancements
python programming_query_cli.py -q "React hooks" --format json --web
"""
    )

    parser.add_argument('--query', '-q', type=str, help='Single query to execute')
    parser.add_argument('--batch', '-b', type=str, help='File containing queries (one per line)')
    parser.add_argument('--output', '-o', type=str, help='Output file for batch results')
    parser.add_argument('--format', '-f', choices=['rich', 'json'], default='rich', help='Output format')
    parser.add_argument('--top-k', '-k', type=int, default=15, help='Number of final results')
    parser.add_argument('--threshold', '-t', type=float, default=0.6, help='Similarity score threshold')

    # New enhanced arguments
    parser.add_argument('--web', action='store_true', help='Enable web search integration')
    parser.add_argument('--model', type=str, choices=list(PROGRAMMING_EMBEDDING_MODELS.keys()),
                        help='Choose embedding model type')
    parser.add_argument('--no-memory', action='store_true', help='Disable enhanced memory features')

    args = parser.parse_args()

    # Create enhanced CLI instance
    cli = EnhancedProgrammingQueryCLI()

    # Update configuration with new arguments
    if args.top_k:
        cli.config.final_top_k = args.top_k
    if args.threshold:
        cli.config.score_threshold = args.threshold
    if args.web:
        cli.config.enable_web_search = True
    if args.model:
        cli.config.embedding_model_type = args.model
        cli.engine.switch_embedding_model(args.model)
    if args.no_memory:
        cli.config.enhanced_memory = False

    try:
        if args.batch:
            return cli.run_batch(args.batch, args.output)  # You'll need to implement this
        elif args.query:
            return cli.run_single_query(args.query, args.format)
        else:
            cli.run_interactive()
        return 0
    except KeyboardInterrupt:
        console.print("\n👋 Operation cancelled by user.", style="dim")
        return 1
    except Exception as e:
        console.print(f"❌ Fatal error: {e}", style="bold red")
        return 1


if __name__ == "__main__":
    sys.exit(main())
