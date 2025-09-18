from dotenv import load_dotenv
from os import getenv
from openai import OpenAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from typing import List, Dict, Optional, Literal
from tavily import TavilyClient
import json
import re
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize clients
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=getenv("OPENROUTER_API_KEY"),
)

tavily_client = TavilyClient(api_key=getenv("TAVILY_API_KEY"))

# Pydantic models for structured data
class Claim(BaseModel):
    claim_text: str
    claim_type: str  # e.g., "statistic", "quote", "event", "policy", "scientific_fact"
    entities: Dict[str, str]  # people, organizations, locations, dates
    verifiability: Literal["high", "medium", "low"]  # how verifiable this claim is
    context: str  # surrounding context from the article

class Evidence(BaseModel):
    source: str
    summary: str
    credibility_score: float  # 0-1 scale based on source reputation
    publication_date: Optional[str] = None
    relevance_score: float  # how relevant this evidence is to the claim
    stance: Literal["supports", "contradicts", "neutral"]

class FactCheckResult(BaseModel):
    claim: str
    context: str
    evidence: List[Evidence]
    verdict: Literal["True", "False", "Mostly True", "Mostly False", "Unverified", "Misleading"]
    confidence_score: Literal["High", "Medium", "Low"]
    reasoning: str
    sources_count: int
    credible_sources_count: int

class NewsFactCheckState(BaseModel):
    input_text: str
    extracted_claims: List[Claim] = []
    evidence_results: Dict[int, List[Evidence]] = {}
    fact_check_results: List[FactCheckResult] = []
    error_messages: List[str] = []

def extract_claims_agent(state: NewsFactCheckState) -> NewsFactCheckState:
    """
    Extract verifiable claims from news articles or statements
    """
    print("🔍 Starting News Claim Extraction...")
    
    # Simplified, more direct extraction prompt
    extraction_prompt = f"""Extract factual claims that can be verified from this text. Look for specific facts, numbers, quotes, dates, and events - NOT opinions.

Text: {state.input_text}

Find claims like:
- "Unemployment is 3.2%"  
- "Biden said X"
- "GDP grew by 2.1%"
- "500,000 jobs were created"

Return ONLY a JSON array like this:
[
  {{
    "claim_text": "unemployment has dropped to 3.2%",
    "claim_type": "statistic", 
    "entities": {{"metric": "unemployment", "value": "3.2%", "timeframe": "current"}},
    "verifiability": "high",
    "context": "President Biden announced today that unemployment has dropped to 3.2%"
  }}
]

Extract ALL verifiable factual claims you can find."""
    
    try:
        response = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3.1:free",
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0.0,  # More deterministic
            max_tokens=1500
        )
        
        claims_text = response.choices[0].message.content.strip()
        print(f"📝 Raw response: {claims_text}")
        
        # Try to extract and clean JSON
        claims = []
        json_match = re.search(r'\[.*\]', claims_text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(0)
            print(f"🔧 Extracted JSON: {json_str[:200]}...")
            
            try:
                claims_data = json.loads(json_str)
                print(f"📊 Parsed {len(claims_data)} potential claims")
                
                for i, claim_dict in enumerate(claims_data):
                    try:
                        # Ensure all required fields exist
                        if not claim_dict.get('claim_text'):
                            print(f"   ⚠️ Skipping claim {i+1}: No claim_text")
                            continue
                            
                        # Set defaults for missing fields
                        claim_dict.setdefault('claim_type', 'other')
                        claim_dict.setdefault('entities', {})
                        claim_dict.setdefault('verifiability', 'medium')
                        claim_dict.setdefault('context', claim_dict['claim_text'])
                        
                        claim = Claim(**claim_dict)
                        claims.append(claim)
                        print(f"   ✅ Added claim {i+1}: {claim.claim_text[:60]}...")
                        
                    except Exception as e:
                        print(f"   ⚠️ Skipping malformed claim {i+1}: {e}")
                        continue
                        
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed: {e}")
                print("🔧 Trying manual extraction...")
                
                # Fallback: extract claims manually from response
                lines = claims_text.split('\n')
                for line in lines:
                    if any(keyword in line.lower() for keyword in ['unemployment', 'gdp', 'jobs', 'inflation', '%', 'said', 'announced']):
                        if len(line.strip()) > 10:
                            claim = Claim(
                                claim_text=line.strip(),
                                claim_type="extracted",
                                entities={},
                                verifiability="medium", 
                                context=line.strip()
                            )
                            claims.append(claim)
                
        else:
            print("❌ No JSON found - trying text analysis...")
            # Fallback: analyze text directly for obvious claims
            text_lower = state.input_text.lower()
            sentences = state.input_text.split('.')
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 10:
                    continue
                    
                # Look for obvious factual patterns
                if any(pattern in sentence.lower() for pattern in [
                    '%', 'percent', 'million', 'billion', 'trillion',
                    'said', 'stated', 'announced', 'reported',
                    'increased', 'decreased', 'grew', 'dropped'
                ]):
                    claim = Claim(
                        claim_text=sentence,
                        claim_type="pattern_match",
                        entities={},
                        verifiability="medium",
                        context=sentence
                    )
                    claims.append(claim)
                    
        print(f"✅ Final result: {len(claims)} claims extracted")
        for i, claim in enumerate(claims, 1):
            print(f"   {i}. [{claim.claim_type}] {claim.claim_text[:80]}...")
            
    except Exception as e:
        print(f"❌ Error in claim extraction: {e}")
        state.error_messages.append(f"Claim extraction error: {e}")
        claims = []
    
    state.extracted_claims = claims
    return state

def evidence_search_agent(state: NewsFactCheckState) -> NewsFactCheckState:
    """
    Search for evidence for each extracted claim
    """
    print(f"🔎 Starting Evidence Search for {len(state.extracted_claims)} claims...")
    
    if not state.extracted_claims:
        print("⚠️ No claims to search evidence for")
        return state
    
    evidence_results = {}
    
    for i, claim in enumerate(state.extracted_claims):
        print(f"   📰 Claim {i+1}/{len(state.extracted_claims)}: {claim.claim_text[:60]}...")
        
        # Create multiple search queries for better coverage
        search_queries = []
        
        # Main query based on claim content
        main_entities = []
        for key, value in claim.entities.items():
            if value and value.lower() not in ['', 'n/a', 'none', 'not specified']:
                main_entities.append(value)
        
        if main_entities:
            primary_query = " ".join(main_entities[:3])  # Use top 3 entities
        else:
            # Extract key terms from claim text
            words = claim.claim_text.split()
            primary_query = " ".join([w for w in words if len(w) > 3][:5])
        
        search_queries.append(primary_query)
        
        # Add fact-check specific query
        if claim.claim_type in ["statistic", "economic_data", "demographic"]:
            search_queries.append(f"{primary_query} data statistics")
        elif claim.claim_type == "quote":
            person = claim.entities.get("person", "")
            if person:
                search_queries.append(f'"{person}" quote statement')
        
        print(f"     🔍 Search queries: {search_queries}")
        
        evidence_list = []
        
        for query in search_queries[:2]:  # Limit to 2 queries per claim
            try:
                search_results = tavily_client.search(
                    query=query,
                    search_depth="advanced",
                    max_results=4,
                    include_domains=["reuters.com", "apnews.com", "bbc.com", "cnn.com", 
                                   "nytimes.com", "washingtonpost.com", "npr.org", 
                                   "factcheck.org", "snopes.com", "politifact.com"]
                )
                
                for result in search_results.get('results', []):
                    # Evaluate relevance and credibility
                    domain = result.get('url', '').split('/')[2] if result.get('url') else 'unknown'
                    
                    # Credibility scoring based on source
                    credibility_score = 0.5  # default
                    if any(trusted in domain.lower() for trusted in 
                          ['reuters', 'ap.org', 'apnews', 'bbc', 'npr']):
                        credibility_score = 0.9
                    elif any(news in domain.lower() for news in 
                            ['nytimes', 'washingtonpost', 'cnn', 'guardian']):
                        credibility_score = 0.8
                    elif any(fact in domain.lower() for fact in 
                            ['factcheck', 'snopes', 'politifact']):
                        credibility_score = 0.95
                    elif domain.endswith('.gov') or domain.endswith('.edu'):
                        credibility_score = 0.85
                    
                    content = result.get('content', '')[:1500]  # Limit content length
                    
                    # Determine relevance and stance
                    relevance_prompt = f"""
                    Analyze how this search result relates to the claim:
                    
                    CLAIM: {claim.claim_text}
                    
                    SEARCH RESULT:
                    Title: {result.get('title', 'No title')}
                    Content: {content}
                    
                    Provide:
                    1. Summary of relevant information (2-3 sentences max)
                    2. Relevance score (0.0-1.0) - how well this addresses the claim
                    3. Stance: does this "supports", "contradicts", or is "neutral" to the claim?
                    
                    Return JSON:
                    {{
                        "summary": "brief relevant summary",
                        "relevance_score": 0.7,
                        "stance": "supports/contradicts/neutral"
                    }}
                    """
                    
                    try:
                        relevance_response = client.chat.completions.create(
                            model="deepseek/deepseek-chat-v3.1:free",
                            messages=[{"role": "user", "content": relevance_prompt}],
                            temperature=0.1,
                            max_tokens=300
                        )
                        
                        relevance_text = relevance_response.choices[0].message.content
                        json_match = re.search(r'\{[\s\S]*\}', relevance_text)
                        
                        if json_match:
                            relevance_data = json.loads(json_match.group())
                            
                            # Only include if relevance is reasonable
                            if relevance_data.get('relevance_score', 0) > 0.3:
                                evidence = Evidence(
                                    source=f"{result.get('title', 'Unknown')} ({domain})",
                                    summary=relevance_data.get('summary', 'No summary available'),
                                    credibility_score=credibility_score,
                                    relevance_score=relevance_data.get('relevance_score', 0.5),
                                    stance=relevance_data.get('stance', 'neutral')
                                )
                                evidence_list.append(evidence)
                                
                    except Exception as e:
                        print(f"       ⚠️ Error analyzing result: {e}")
                        continue
                        
            except Exception as e:
                print(f"     ❌ Search error: {e}")
                continue
        
        # Sort by relevance and credibility
        evidence_list.sort(key=lambda x: x.relevance_score * x.credibility_score, reverse=True)
        evidence_results[i] = evidence_list[:5]  # Keep top 5 pieces of evidence
        
        print(f"     ✅ Found {len(evidence_list)} relevant sources")
        
    state.evidence_results = evidence_results
    return state

def fact_check_agent(state: NewsFactCheckState) -> NewsFactCheckState:
    """
    Evaluate each claim against gathered evidence
    """
    print(f"⚖️ Starting Fact-Check Analysis for {len(state.extracted_claims)} claims...")
    
    if not state.extracted_claims:
        print("⚠️ No claims to fact-check")
        return state
    
    fact_check_results = []
    
    for i, claim in enumerate(state.extracted_claims):
        evidence_list = state.evidence_results.get(i, [])
        print(f"   ⚖️ Analyzing claim {i+1}: {claim.claim_text[:50]}...")
        
        # Prepare evidence summary for analysis
        evidence_summary = ""
        supporting_count = len([e for e in evidence_list if e.stance == "supports"])
        contradicting_count = len([e for e in evidence_list if e.stance == "contradicts"])
        credible_sources = len([e for e in evidence_list if e.credibility_score > 0.7])
        
        for j, evidence in enumerate(evidence_list, 1):
            evidence_summary += f"{j}. [{evidence.stance.upper()}] {evidence.summary}\n"
            evidence_summary += f"   Source: {evidence.source} (Credibility: {evidence.credibility_score:.2f})\n\n"
        
        fact_check_prompt = f"""
        As an expert fact-checker, evaluate this claim against the available evidence:
        
        CLAIM: {claim.claim_text}
        CLAIM TYPE: {claim.claim_type}
        CONTEXT: {claim.context}
        ENTITIES: {claim.entities}
        
        EVIDENCE SUMMARY:
        Total Sources: {len(evidence_list)}
        Supporting: {supporting_count}
        Contradicting: {contradicting_count}
        Credible Sources (>0.7): {credible_sources}
        
        EVIDENCE DETAILS:
        {evidence_summary}
        
        Based on this evidence, determine:
        
        1. VERDICT: Choose the most appropriate:
           - "True": Strong evidence supports the claim
           - "Mostly True": Generally accurate with minor issues
           - "False": Strong evidence contradicts the claim  
           - "Mostly False": Generally inaccurate with some truth
           - "Misleading": Technically accurate but missing important context
           - "Unverified": Insufficient evidence to determine truth
        
        2. CONFIDENCE: "High", "Medium", or "Low" based on:
           - Quality and quantity of evidence
           - Source credibility
           - Consistency across sources
        
        3. REASONING: Detailed explanation (3-4 sentences) covering:
           - Key evidence that informed your decision
           - Why you chose this verdict and confidence level
           - Any important caveats or context
        
        Return JSON:
        {{
            "verdict": "True/False/Mostly True/Mostly False/Misleading/Unverified",
            "confidence_score": "High/Medium/Low", 
            "reasoning": "detailed explanation of decision"
        }}
        """
        
        try:
            fact_check_response = client.chat.completions.create(
                model="deepseek/deepseek-chat-v3.1:free",
                messages=[{"role": "user", "content": fact_check_prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            fact_check_text = fact_check_response.choices[0].message.content
            json_match = re.search(r'\{[\s\S]*\}', fact_check_text)
            
            if json_match:
                fact_check_data = json.loads(json_match.group())
                
                result = FactCheckResult(
                    claim=claim.claim_text,
                    context=claim.context,
                    evidence=evidence_list,
                    verdict=fact_check_data.get('verdict', 'Unverified'),
                    confidence_score=fact_check_data.get('confidence_score', 'Low'),
                    reasoning=fact_check_data.get('reasoning', 'Unable to determine'),
                    sources_count=len(evidence_list),
                    credible_sources_count=credible_sources
                )
                
            else:
                result = FactCheckResult(
                    claim=claim.claim_text,
                    context=claim.context,
                    evidence=evidence_list,
                    verdict="Unverified",
                    confidence_score="Low",
                    reasoning="Error parsing fact-check analysis",
                    sources_count=len(evidence_list),
                    credible_sources_count=credible_sources
                )
                
        except Exception as e:
            print(f"     ❌ Error in fact-checking: {e}")
            result = FactCheckResult(
                claim=claim.claim_text,
                context=claim.context,
                evidence=evidence_list,
                verdict="Unverified",
                confidence_score="Low",
                reasoning=f"Technical error in analysis: {str(e)[:100]}",
                sources_count=len(evidence_list),
                credible_sources_count=credible_sources
            )
        
        fact_check_results.append(result)
        print(f"     📊 Verdict: {result.verdict} ({result.confidence_score} confidence)")
    
    state.fact_check_results = fact_check_results
    return state

def create_news_fact_checking_workflow():
    """
    Create the news fact-checking workflow with proper error handling
    """
    workflow = StateGraph(NewsFactCheckState)
    
    # Add all agent nodes
    workflow.add_node("extract_claims", extract_claims_agent)
    workflow.add_node("search_evidence", evidence_search_agent)  
    workflow.add_node("fact_check", fact_check_agent)
    
    # Set up the flow
    workflow.add_edge(START, "extract_claims")
    workflow.add_edge("extract_claims", "search_evidence")
    workflow.add_edge("search_evidence", "fact_check")
    workflow.add_edge("fact_check", END)
    
    return workflow.compile()

def format_news_results(results: List[FactCheckResult]) -> str:
    """
    Format results for news fact-checking display
    """
    if not results:
        return "\n❌ No claims were extracted or analyzed.\n"
    
    output = "\n" + "="*80 + "\n"
    output += "📰 NEWS FACT-CHECK RESULTS\n"
    output += "="*80 + "\n\n"
    
    # Verdict emoji mapping
    verdict_emojis = {
        "True": "✅",
        "Mostly True": "✅", 
        "False": "❌",
        "Mostly False": "❌",
        "Misleading": "⚠️",
        "Unverified": "❓"
    }
    
    for i, result in enumerate(results, 1):
        emoji = verdict_emojis.get(result.verdict, "❓")
        
        output += f"{emoji} CLAIM {i}: {result.verdict.upper()}\n"
        output += f"Statement: \"{result.claim}\"\n"
        output += f"Context: {result.context[:200]}{'...' if len(result.context) > 200 else ''}\n"
        output += f"Confidence: {result.confidence_score}\n"
        output += f"Sources Analyzed: {result.sources_count} ({result.credible_sources_count} credible)\n"
        output += f"Analysis: {result.reasoning}\n"
        
        if result.evidence:
            output += f"\nKey Evidence:\n"
            for j, evidence in enumerate(result.evidence[:3], 1):  # Show top 3
                stance_emoji = "✅" if evidence.stance == "supports" else "❌" if evidence.stance == "contradicts" else "➖"
                output += f"  {stance_emoji} {evidence.summary}\n"
                output += f"     Source: {evidence.source}\n"
        
        output += "\n" + "─"*60 + "\n\n"
    
    # Summary statistics
    verdict_counts = {}
    for result in results:
        verdict_counts[result.verdict] = verdict_counts.get(result.verdict, 0) + 1
    
    output += "📊 SUMMARY STATISTICS:\n"
    output += f"Total Claims Analyzed: {len(results)}\n"
    for verdict, count in verdict_counts.items():
        emoji = verdict_emojis.get(verdict, "❓")
        output += f"{emoji} {verdict}: {count}\n"
    
    return output

def export_news_results_json(results: List[FactCheckResult]) -> str:
    """
    Export results in structured JSON format for news fact-checking
    """
    json_results = []
    for result in results:
        json_result = {
            "claim": result.claim,
            "context": result.context,
            "verdict": result.verdict,
            "confidence_score": result.confidence_score,
            "reasoning": result.reasoning,
            "evidence_summary": {
                "total_sources": result.sources_count,
                "credible_sources": result.credible_sources_count,
                "sources": [
                    {
                        "source": evidence.source,
                        "summary": evidence.summary,
                        "stance": evidence.stance,
                        "credibility_score": evidence.credibility_score,
                        "relevance_score": evidence.relevance_score
                    } for evidence in result.evidence
                ]
            }
        }
        json_results.append(json_result)
    
    return json.dumps(json_results, indent=2)

# Main execution
if __name__ == "__main__":
    print("🚀 Advanced News Fact-Checking System")
    print("=" * 50)
    
    # Sample news text for testing
    sample_news = """
    President Biden announced today that unemployment has dropped to 3.2%, the lowest in 50 years. 
    The Bureau of Labor Statistics reported that 500,000 new jobs were created last month alone.
    "This shows our economic policies are working," Biden stated during a White House press conference.
    The GDP has grown by 2.1% this quarter, exceeding economists' predictions of 1.8% growth.
    Meanwhile, inflation remains at 4.2%, down from last year's peak of 9.1%.
    """
    
    print("\nChoose input method:")
    print("1. Enter your own news text")
    print("2. Use sample news article")
    print("3. Test claim extraction only")
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        user_input = input("\nPaste the news article or statement to fact-check:\n").strip()
        if not user_input:
            print("No input provided. Using sample text.")
            user_input = sample_news
    elif choice == "3":
        # Test extraction only
        print(f"\nTesting claim extraction with sample text:\n{sample_news}\n")
        test_state = NewsFactCheckState(input_text=sample_news)
        test_result = extract_claims_agent(test_state)
        print(f"\n✅ Extraction test complete. Found {len(test_result.extracted_claims)} claims.")
        if test_result.extracted_claims:
            print("\nExtracted claims:")
            for i, claim in enumerate(test_result.extracted_claims, 1):
                print(f"{i}. {claim.claim_text}")
        exit()
    else:
        user_input = sample_news
        print(f"\nUsing sample news text:\n{sample_news}\n")
    
    # Create and run the workflow
    try:
        app = create_news_fact_checking_workflow()
        initial_state = NewsFactCheckState(input_text=user_input)
        
        print("🔄 Running News Fact-Checking Pipeline...\n")
        result = app.invoke(initial_state)
        
        # Display any error messages
        if hasattr(result, 'error_messages') and result.error_messages:
            print("⚠️ Warnings/Errors:")
            for error in result.error_messages:
                print(f"  - {error}")
            print()
        
        # Get results - handle both dict and object returns
        if isinstance(result, dict):
            fact_check_results = result.get('fact_check_results', [])
        else:
            fact_check_results = result.fact_check_results
        # Display formatted results
        formatted_output = format_news_results(fact_check_results)
        print(formatted_output)
        
        # Ask if user wants JSON export
        if fact_check_results:
            export_choice = input("Export detailed results as JSON? (y/n): ").lower().strip()
            if export_choice == 'y':
                json_output = export_news_results_json(fact_check_results)
                
                # Save to file option
                save_choice = input("Save to file? (y/n): ").lower().strip()
                if save_choice == 'y':
                    filename = f"fact_check_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(filename, 'w') as f:
                        f.write(json_output)
                    print(f"Results saved to {filename}")
                else:
                    print("\n" + "="*50)
                    print("JSON EXPORT:")
                    print("="*50)
                    print(json_output)
                    
    except Exception as e:
        print(f"❌ Error running fact-checking pipeline: {e}")
        import traceback
        traceback.print_exc()