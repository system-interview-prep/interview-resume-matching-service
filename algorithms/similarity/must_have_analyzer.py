import re
from ..base_algorithm import BaseAlgorithm


class MustHaveAnalyzer(BaseAlgorithm):
    """Rule-based must-have coverage analyzer that does not depend on spaCy."""

    def __init__(self, config: dict = None):
        super().__init__('must_have', config)
        self.skill_terms = self._load_skill_terms()
        self.must_markers = [
            'must have', 'must-have', 'required', 'requirements', 'requirement',
            'mandatory', 'essential', 'bắt buộc', 'yeu cau', 'yêu cầu',
        ]
        self.stop_markers = [
            'nice to have', 'nice-to-have', 'preferred', 'bonus', 'plus',
            'ưu tiên', 'uu tien', 'benefits', 'phúc lợi',
        ]

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
            'microservices', 'ci/cd', 'cicd', 'linux',
        }

    def _normalize(self, text: str) -> str:
        text = (text or '').lower()
        replacements = {
            'node.js': 'nodejs',
            'next.js': 'nextjs',
            'reactjs': 'react',
            'vuejs': 'vue',
            'cicd': 'ci/cd',
        }
        for src, dst in replacements.items():
            text = text.replace(src, dst)
        return text

    def _contains_term(self, text: str, term: str) -> bool:
        if term in {'c++', 'c#', 'ci/cd'}:
            return term in text
        pattern = r'(?<![a-z0-9+#./-])' + re.escape(term) + r'(?![a-z0-9+#./-])'
        return re.search(pattern, text, re.IGNORECASE) is not None

    def _extract_must_section(self, job_text: str) -> str:
        text = self._normalize(job_text)
        starts = [text.find(marker) for marker in self.must_markers if text.find(marker) >= 0]
        if not starts:
            return text

        start = min(starts)
        stops = [text.find(marker, start + 1) for marker in self.stop_markers if text.find(marker, start + 1) >= 0]
        end = min(stops) if stops else len(text)
        return text[start:end]

    def _extract_required_skills(self, job_text: str) -> list:
        section = self._extract_must_section(job_text)
        found = [term for term in sorted(self.skill_terms) if self._contains_term(section, term)]
        return found

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

    def process_single(self, resume_text: str, job_description: str, position: str = None) -> dict:
        if not self.is_loaded:
            self.load_model()

        resume = self._normalize(resume_text)
        required_skills = self._extract_required_skills(job_description)
        missing_skills = [skill for skill in required_skills if not self._contains_term(resume, skill)]

        required_years = self._extract_year_requirement(job_description)
        resume_years = self._extract_resume_years(resume_text)
        experience_ok = required_years == 0 or resume_years >= required_years

        checks = len(required_skills) + (1 if required_years else 0)
        missing_count = len(missing_skills) + (0 if experience_ok else 1)
        coverage = 1.0 if checks == 0 else max(0.0, 1.0 - (missing_count / checks))
        must_have_ok = missing_count == 0

        return {
            'algorithm': self.name,
            'score': float(coverage),
            'details': {
                'required_skills': required_skills,
                'missing_must_have': missing_skills,
                'missing_must_count': missing_count,
                'must_have_ok': must_have_ok,
                'required_years_experience': required_years,
                'resume_years_experience': resume_years,
                'experience_ok': experience_ok,
            },
        }
