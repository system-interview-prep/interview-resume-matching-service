import re
from ..base_algorithm import BaseAlgorithm


class RequirementsAnalyzer(BaseAlgorithm):
    """Structured requirements analyzer for must-have, nice-to-have, and constraints."""

    def __init__(self, config: dict = None):
        super().__init__('requirements', config)
        self.must_weight = float(self.config.get('must_have_weight', 0.55))
        self.nice_weight = float(self.config.get('nice_to_have_weight', 0.25))
        self.constraints_weight = float(self.config.get('constraints_weight', 0.20))
        self.skill_terms = self._load_skill_terms()

    def load_model(self):
        self.is_loaded = True

    def _load_skill_terms(self):
        return {
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust',
            'php', 'ruby', 'swift', 'kotlin', 'scala', 'r', 'matlab',
            'react', 'angular', 'vue', 'nextjs', 'next.js', 'nestjs', 'nodejs',
            'node.js', 'express', 'django', 'flask', 'spring', 'laravel',
            'fastapi', 'tensorflow', 'pytorch', 'scikit-learn',
            'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
            'oracle', 'sqlite', 'dynamodb', 'sql', 'nosql',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins',
            'terraform', 'ansible', 'git', 'jira', 'figma',
            'html', 'css', 'tailwind', 'bootstrap', 'rest', 'graphql',
            'microservices', 'ci/cd', 'cicd', 'linux', 'english', 'vietnamese',
            'spark', 'pyspark', 'json', 'pandas', 'dataframe', 'dataframes',
            'data warehouse', 'data warehousing', 'data lake', 'data mart',
            'data marts', 'etl', 'elt', 'data pipeline', 'data pipelines',
            'hadoop', 'hive', 'kafka', 'mapreduce', 'looker', 'powerbi',
            'power bi', 'gitlab', 'confluence', 'kanban', 'agile',
            'data architecture', 'data engineering', 'data science',
            'machine learning', 'ai', 'metadata management',
            'data governance', 'master data management', 'data catalog',
            'business glossary', 'computer science', 'algorithms',
            'data structures', 'columnar databases',
        }

    def _normalize(self, text: str) -> str:
        text = (text or '').lower()
        replacements = {
            'node.js': 'nodejs',
            'next.js': 'nextjs',
            'reactjs': 'react',
            'vuejs': 'vue',
            'cicd': 'ci/cd',
            'power bi': 'powerbi',
            'dataframes': 'dataframe',
        }
        for src, dst in replacements.items():
            text = text.replace(src, dst)
        return text

    def _contains_term(self, text: str, term: str) -> bool:
        term = self._normalize(term).strip()
        if not term:
            return False
        if term in {'c++', 'c#', 'ci/cd'}:
            return term in text
        if len(term.split()) > 1:
            return term in text
        pattern = r'(?<![a-z0-9+#/-])' + re.escape(term) + r'(?![a-z0-9+#/-])'
        return re.search(pattern, text, re.IGNORECASE) is not None

    def _extract_section(self, text: str, title_patterns: list) -> str:
        lines = (text or '').splitlines()
        capture = False
        out = []
        known_headers = [
            'must have', 'must-have', 'required', 'requirements', 'nice to have',
            'nice-to-have', 'preferred', 'constraints', 'constraint', 'responsibilities',
            'tech stack', 'skills',
        ]

        for line in lines:
            clean = line.strip().lower().rstrip(':')
            if any(re.fullmatch(pattern, clean) for pattern in title_patterns):
                capture = True
                continue
            if capture and not clean and out:
                break
            if capture and any(clean == header for header in known_headers):
                break
            if capture:
                out.append(line)

        return '\n'.join(out).strip()

    def _bullet_items(self, section: str) -> list:
        raw_lines = (section or '').splitlines()
        has_bullets = any(raw.lstrip().startswith(('-', '+', '*')) for raw in raw_lines)
        items = []
        for raw in raw_lines:
            if has_bullets and not raw.lstrip().startswith(('-', '+', '*')):
                continue
            line = re.sub(r'^\s*[-+*•]\s*', '', raw).strip()
            if line:
                items.append(line)
        return items

    def _canonicalize_item(self, item: str) -> str:
        normalized = self._normalize(item)
        matched_terms = [term for term in sorted(self.skill_terms) if self._contains_term(normalized, term)]
        if len(matched_terms) == 1:
            return matched_terms[0]
        return item.strip()

    def _terms_in_text(self, text: str) -> list:
        normalized = self._normalize(text)
        return [term for term in sorted(self.skill_terms) if self._contains_term(normalized, term)]

    def _soft_item_score(self, resume: str, item: str) -> tuple:
        """Score a requirement item by its meaningful terms instead of exact sentence match."""
        item_norm = self._normalize(item)
        if not item_norm.strip():
            return 0.0, []

        if self._contains_term(resume, item_norm):
            return 1.0, [item]

        terms = self._terms_in_text(item_norm)
        matched_terms = [term for term in terms if self._contains_term(resume, term)]
        if terms:
            return len(matched_terms) / len(terms), matched_terms

        heuristics = []
        if any(x in item_norm for x in ['team', 'manage', 'manager', 'lead', 'leadership']):
            leadership_terms = ['lead', 'leader', 'manager', 'manage', 'team', 'mentor', 'supervise']
            hit = [x for x in leadership_terms if self._contains_term(resume, x)]
            heuristics.append(1.0 if hit else 0.0)
            matched_terms.extend(hit)
        if any(x in item_norm for x in ['bachelor', 'master', 'degree', 'computer science']):
            degree_terms = ['bachelor', 'master', 'degree', 'computer science', 'computer engineering', 'engineering']
            hit = [x for x in degree_terms if self._contains_term(resume, x)]
            heuristics.append(min(1.0, len(hit) / 2))
            matched_terms.extend(hit)
        if any(x in item_norm for x in ['communication', 'verbal', 'written']):
            comm_terms = ['communication', 'communicate', 'written', 'verbal']
            hit = [x for x in comm_terms if self._contains_term(resume, x)]
            heuristics.append(1.0 if hit else 0.0)
            matched_terms.extend(hit)

        if heuristics:
            return max(heuristics), sorted(set(matched_terms))

        return 0.0, []

    def _extract_items(self, job_description: str, kind: str) -> list:
        text = job_description or ''
        if kind == 'must':
            section = self._extract_section(text, [r'must\s*have', r'must-have', r'required', r'requirements'])
        elif kind == 'nice':
            section = self._extract_section(text, [r'nice\s*to\s*have', r'nice-to-have', r'preferred'])
        else:
            section = self._extract_section(text, [r'constraints?', r'constraint'])

        items = self._bullet_items(section)
        if items:
            return [self._canonicalize_item(item) for item in items]

        if kind != 'must':
            return []

        normalized = self._normalize(text)
        return [term for term in sorted(self.skill_terms) if self._contains_term(normalized, term)]

    def _extract_year_requirement(self, job_text: str) -> int:
        text = self._normalize(job_text)
        patterns = [
            r'(\d+)\s*\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',
            r'(?:at least|minimum|min)\s*(\d+)\s*(?:years?|yrs?)',
            r'tối thiểu\s*(\d+)\s*năm',
            r'ít nhất\s*(\d+)\s*năm',
        ]
        years = []
        for pattern in patterns:
            years.extend(int(x) for x in re.findall(pattern, text) if str(x).isdigit())
        return max(years) if years else 0

    def _extract_resume_years(self, resume_text: str) -> int:
        text = self._normalize(resume_text)
        patterns = [
            r'(\d+)\s*\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',
            r'(\d+)\s*\+?\s*(?:years?|yrs?)',
            r'(\d+)\s*\+?\s*năm\s*(?:kinh nghiệm)?',
        ]
        years = []
        for pattern in patterns:
            years.extend(int(x) for x in re.findall(pattern, text) if str(x).isdigit())
        return max(years) if years else 0

    def _coverage(self, resume_text: str, items: list) -> tuple:
        resume = self._normalize(resume_text)
        matched = []
        missing = []
        partials = []
        for item in items:
            item_score, matched_terms = self._soft_item_score(resume, item)
            partials.append(item_score)
            if item_score >= 0.5:
                matched.append(item if not matched_terms else f"{item} ({', '.join(matched_terms[:6])})")
            else:
                missing.append(item)
        score = 1.0 if not items else sum(partials) / len(items)
        return float(score), matched, missing

    def process_single(self, resume_text: str, job_description: str, position: str = None) -> dict:
        if not self.is_loaded:
            self.load_model()

        must_items = self._extract_items(job_description, 'must')
        nice_items = self._extract_items(job_description, 'nice')
        constraint_items = self._extract_items(job_description, 'constraints')

        must_score, matched_must, missing_must = self._coverage(resume_text, must_items)
        nice_score, matched_nice, missing_nice = self._coverage(resume_text, nice_items)
        constraints_score, matched_constraints, missing_constraints = self._coverage(resume_text, constraint_items)

        required_years = self._extract_year_requirement(job_description)
        resume_years = self._extract_resume_years(resume_text)
        experience_ok = required_years == 0 or resume_years >= required_years
        if required_years and not experience_ok:
            if resume_years > 0:
                missing_must = missing_must + [f'{required_years}+ years experience']
                must_score = must_score * 0.75
            else:
                # Many resumes omit an explicit "X years" phrase even when experience exists.
                # Treat unknown tenure as weak evidence, not a hard must-have miss.
                must_score = must_score * 0.90

        total_weight = self.must_weight + self.nice_weight + self.constraints_weight
        score = (
            must_score * self.must_weight
            + nice_score * self.nice_weight
            + constraints_score * self.constraints_weight
        ) / total_weight if total_weight > 0 else 0.0

        must_have_ok = len(missing_must) == 0
        constraints_ok = len(missing_constraints) == 0

        return {
            'algorithm': self.name,
            'score': float(max(0.0, min(1.0, score))),
            'details': {
                'must_have_score': float(must_score),
                'nice_to_have_score': float(nice_score),
                'constraints_score': float(constraints_score),
                'must_have': must_items,
                'matched_must_have': matched_must,
                'missing_must_have': missing_must,
                'missing_must_count': len(missing_must),
                'must_have_ok': must_have_ok,
                'nice_to_have': nice_items,
                'matched_nice_to_have': matched_nice,
                'missing_nice_to_have': missing_nice,
                'constraints': constraint_items,
                'matched_constraints': matched_constraints,
                'missing_constraints': missing_constraints,
                'constraints_ok': constraints_ok,
                'required_years_experience': required_years,
                'resume_years_experience': resume_years,
                'experience_ok': experience_ok,
                'weights': {
                    'must_have': self.must_weight,
                    'nice_to_have': self.nice_weight,
                    'constraints': self.constraints_weight,
                },
            },
        }
