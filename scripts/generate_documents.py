from groq import Groq
import json
import os
import time

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY = "temp"  
OUTPUT_DIR = "data"
DOCS_PER_DOMAIN = 30

client = Groq(api_key=API_KEY)

# ── Domain definitions ─────────────────────────────────────────────────────────
DOMAINS = {
    # "HR_Policies": [
    #     "remote work policy", "leave and time-off policy", "code of conduct",
    #     "performance review policy", "hiring and onboarding policy", "expense reimbursement policy",
    #     "anti-harassment policy", "employee benefits policy", "training and development policy",
    #     "disciplinary action policy", "workplace safety policy", "data privacy policy for employees",
    #     "travel policy", "overtime and compensation policy", "promotion and career growth policy",
    #     "termination policy", "social media usage policy", "dress code policy",
    #     "conflict of interest policy", "whistleblower policy", "parental leave policy",
    #     "health and wellness policy", "IT equipment usage policy", "intern policy",
    #     "grievance redressal policy", "equal opportunity policy", "attendance policy",
    #     "relocation policy", "background check policy", "volunteer time off policy"
    # ],
    # "Technical_Documentation": [
    #     "REST API reference guide", "system architecture document", "deployment guide",
    #     "troubleshooting and debugging guide", "database schema documentation",
    #     "microservices integration guide", "CI/CD pipeline setup guide", "security hardening guide",
    #     "load balancing configuration guide", "containerization with Docker guide",
    #     "monitoring and alerting setup guide", "SDK usage documentation",
    #     "authentication and authorization guide", "data migration guide",
    #     "cloud infrastructure setup guide", "caching strategy documentation",
    #     "logging and observability guide", "performance tuning guide",
    #     "disaster recovery plan", "API versioning policy", "service mesh documentation",
    #     "message queue integration guide", "automated testing framework guide",
    #     "configuration management guide", "data pipeline documentation",
    #     "GraphQL API documentation", "WebSocket integration guide",
    #     "rate limiting implementation guide", "secrets management guide",
    #     "infrastructure as code documentation"
    # ],
    # "Financial_Reports": [
    #     "Q1 quarterly earnings report", "Q2 quarterly earnings report",
    #     "Q3 quarterly earnings report", "annual financial report",
    #     "budget forecast document", "expense analysis report",
    #     "revenue breakdown report", "cash flow statement",
    #     "profit and loss statement", "balance sheet report",
    #     "investment portfolio summary", "cost reduction analysis",
    #     "financial risk assessment", "capital expenditure report",
    #     "departmental budget report", "sales performance report",
    #     "accounts receivable summary", "tax compliance report",
    #     "auditor findings report", "financial projections document",
    #     "merger and acquisition financial analysis", "working capital report",
    #     "dividend policy document", "financial compliance report",
    #     "vendor payment analysis", "payroll expense report",
    #     "debt management report", "insurance coverage summary",
    #     "grant utilization report", "financial restatement notice"
    # ],
    "Legal_Documents": [
        "software service agreement", "data privacy policy",
        "terms of service document", "vendor contract agreement",
        "non-disclosure agreement", "employment contract",
        "intellectual property assignment agreement", "liability waiver",
        "lease agreement for office space", "consulting services agreement",
        "partnership agreement", "shareholder agreement",
        "end user license agreement", "data processing agreement",
        "subcontractor agreement", "settlement agreement",
        "letter of intent", "memorandum of understanding",
        "software license agreement", "master service agreement",
        "indemnification agreement", "purchase order terms",
        "freelancer contract", "SLA service level agreement",
        "content licensing agreement", "marketing partnership agreement",
        "affiliate agreement", "customer data agreement",
        "open source contribution agreement", "research collaboration agreement"
    ],
    # "General_Business": [
    #     "project proposal document", "business strategy document",
    #     "meeting minutes report", "company handbook section",
    #     "product roadmap document", "market analysis report",
    #     "competitor analysis document", "customer success plan",
    #     "onboarding guide for new employees", "team OKR document",
    #     "stakeholder communication plan", "change management plan",
    #     "risk management plan", "business continuity plan",
    #     "go-to-market strategy document", "product requirements document",
    #     "user research findings report", "operational efficiency report",
    #     "partnership proposal document", "customer feedback analysis",
    #     "brand guidelines document", "crisis communication plan",
    #     "sustainability report", "diversity and inclusion report",
    #     "quarterly business review document", "sales playbook",
    #     "customer journey map document", "event planning document",
    #     "vendor evaluation report", "internal audit report"
    # ]
}

STRUCTURES = ["prose-heavy", "bullet-heavy", "table-heavy", "mixed"]
LENGTHS = [800, 1000, 1200,1300, 1500]

# ── Prompt ─────────────────────────────────────────────────────────────────────
def build_prompt(domain, doc_type, length, structure):
    return f"""You are generating a synthetic enterprise document and evaluation questions for an AI research dataset.

Domain: {domain.replace('_', ' ')}
Document type: {doc_type}
Target length: {length} words
Structure style: {structure}

Structure styles:
- "prose-heavy": mostly flowing paragraphs, minimal headers
- "bullet-heavy": extensive bullet points and numbered lists
- "table-heavy": multiple tables with realistic data
- "mixed": combination of all above

TASK 1 - Generate the document:
- Use realistic terminology, specific facts, numbers, names (fictional but realistic)
- Follow the specified structure style exactly
- Make it self-contained, detailed, and coherent
- Include specific figures, dates, names, and policies where relevant

TASK 2 - Generate exactly 10 QA pairs from the document you just wrote:
- Each answer must be a direct span or sentence from the document
- Each answer must be fully contained within a single paragraph
- Questions must be specific and factual (not general or yes/no)
- Vary questions across different sections of the document

Return ONLY this JSON with no preamble, no markdown, no code fences:
{{
  "document": "full document text here",
  "qa_pairs": [
    {{
      "question": "...",
      "answer": "...",
      "answer_paragraph": "exact paragraph from document containing the answer"
    }}
  ]
}}"""

# ── Save files ─────────────────────────────────────────────────────────────────
def save_document(domain, index, doc_text, qa_pairs):
    domain_dir = os.path.join(OUTPUT_DIR, domain)
    os.makedirs(domain_dir, exist_ok=True)

    doc_filename = os.path.join(domain_dir, f"doc_{index:02d}.txt")
    qa_filename = os.path.join(domain_dir, f"doc_{index:02d}_qa.json")

    with open(doc_filename, "w", encoding="utf-8") as f:
        f.write(doc_text)

    with open(qa_filename, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Saved {doc_filename} + QA file")

# ── Generate one document ──────────────────────────────────────────────────────
def generate_document(domain, doc_type, length, structure, retries=3):
    prompt = build_prompt(domain, doc_type, length, structure)

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=4096,
                            temperature=0.7,
                            response_format={"type": "json_object"}  # ← add this line
                        )

            raw = response.choices[0].message.content.strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            data = json.loads(raw)
            return data["document"], data["qa_pairs"]

        except json.JSONDecodeError as e:
            print(f"  ⚠ JSON parse error on attempt {attempt+1}: {e}")
            time.sleep(2)

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str.lower():
                wait = 30 * (attempt + 1)
                print(f"  ⚠ Rate limit hit, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  ⚠ API error on attempt {attempt+1}: {e}")
                time.sleep(5)

    print(f"  ✗ Failed after {retries} attempts, skipping.")
    return None, None

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    total = 0
    target = sum(min(len(v), DOCS_PER_DOMAIN) for v in DOMAINS.values())

    for domain, doc_types in DOMAINS.items():
        print(f"\n{'='*60}")
        print(f"Domain: {domain}")
        print(f"{'='*60}")

        selected = doc_types[:DOCS_PER_DOMAIN]

        for i, doc_type in enumerate(selected):
            length = LENGTHS[i % len(LENGTHS)]
            structure = STRUCTURES[i % len(STRUCTURES)]

            print(f"\n[{i+1}/{DOCS_PER_DOMAIN}] Generating: {doc_type} | {length} words | {structure}")

            doc_text, qa_pairs = generate_document(domain, doc_type, length, structure)

            if doc_text and qa_pairs:
                save_document(domain, i+1, doc_text, qa_pairs)
                total += 1

            time.sleep(2)  # Groq: 30 req/min → 2s gap is safe

    print(f"\n{'='*60}")
    print(f"✅ Done! Generated {total}/{target} documents.")
    print(f"📁 Files saved in: {os.path.abspath(OUTPUT_DIR)}/")

if __name__ == "__main__":
    main()