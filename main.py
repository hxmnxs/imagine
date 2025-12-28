import google.generativeai as genai
import base64
import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from datetime import datetime
import hashlib
import os


class ImageType(Enum):
    FLOWCHART = "flowchart"
    SYSTEM_DIAGRAM = "system_diagram"
    PROCESS_FLOW = "process_flow"
    CONCEPTUAL_FRAMEWORK = "conceptual_framework"
    DATA_VISUALIZATION = "data_visualization"
    NETWORK_DIAGRAM = "network_diagram"
    TIMELINE = "timeline"


@dataclass
class ImageGenerationRequest:
    title: str
    description: str
    image_type: ImageType
    complexity_level: int = 2  # 1-5 scale
    style_preferences: Optional[dict] = None
    custom_requirements: Optional[str] = None
    max_retries: int = 3


@dataclass
class GenerationMetrics:
    planning_time: float
    generation_time: float
    validation_time: float
    total_time: float
    refinement_iterations: int
    success: bool


class AcademicImageGenerator:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini API client"""
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY not provided and not found in environment variables")
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.0-flash-exp"
        self.cache = {}
        
    def _get_cache_key(self, request: ImageGenerationRequest) -> str:
        """Generate a consistent cache key for requests"""
        key_str = f"{request.title}_{request.image_type.value}_{request.description[:50]}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _plan_visualization(self, request: ImageGenerationRequest) -> dict:
        """Phase 1: Agent plans the visualization approach"""
        planning_prompt = f"""You are an expert academic visualization designer. Analyze this request and create a detailed visualization plan.

Request Details:
- Title: {request.title}
- Type: {request.image_type.value}
- Complexity Level: {request.complexity_level}/5
- Description: {request.description}
- Custom Requirements: {request.custom_requirements or 'None'}
- Style Preferences: {json.dumps(request.style_preferences or {})}

Create a JSON plan with these sections:
1. "visualization_strategy": How to structure the image for maximum clarity
2. "key_elements": List of essential components to include
3. "layout_approach": Grid/flow/hierarchical/network structure
4. "color_scheme": Recommended academic color palette (with hex codes)
5. "typography_notes": Font families and sizes for hierarchy
6. "technical_approach": Description of layout structure
7. "refinement_criteria": What makes this visualization successful
8. "potential_pitfalls": Common issues to avoid

Return ONLY valid JSON."""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=planning_prompt
            )
            response_text = response.text
            
            # Extract JSON from response
            try:
                plan = json.loads(response_text)
            except json.JSONDecodeError:
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                if start >= 0 and end > start:
                    plan = json.loads(response_text[start:end])
                else:
                    raise ValueError("Failed to parse planning response")
            
            return plan
        except Exception as e:
            raise ValueError(f"Planning phase failed: {str(e)}")
    
    def _generate_image_prompt(self, request: ImageGenerationRequest, plan: dict) -> str:
        """Phase 2: Generate optimized prompt for image creation with Gemini"""
        complexity_multiplier = {
            1: "simple, minimalist",
            2: "clear and moderate detail",
            3: "detailed with good organization",
            4: "comprehensive with multiple layers",
            5: "highly detailed and complex"
        }
        
        prompt = f"""Create a publication-quality academic {request.image_type.value} image with these specifications:

TITLE: {request.title}

REQUIREMENTS:
- Create a {complexity_multiplier.get(request.complexity_level, 'professional')} academic {request.image_type.value}
- Style: Professional academic - suitable for journal papers and presentations
- Format: High quality, clear and readable
- Resolution: Suitable for print quality (300+ DPI equivalent)

DESIGN GUIDELINES:
- Layout approach: {plan.get('layout_approach', 'hierarchical')}
- Color palette: Use {plan.get('color_scheme', {}).get('primary', 'professional colors')} as primary, {plan.get('color_scheme', {}).get('secondary', 'complementary colors')} as secondary
- Typography: {plan.get('typography_notes', 'Clear sans-serif fonts with proper hierarchy')}
- Key elements to include: {', '.join(plan.get('key_elements', []))}

STRUCTURE:
- Visualization strategy: {plan.get('visualization_strategy', 'clear and organized')}
- Technical approach: {plan.get('technical_approach', 'standard academic format')}

QUALITY CRITERIA - Ensure the image has these characteristics:
{chr(10).join([f'• {criterion}' for criterion in plan.get('refinement_criteria', [])])}

THINGS TO AVOID:
{chr(10).join([f'• {pitfall}' for pitfall in plan.get('potential_pitfalls', [])])}

CONTEXT: {request.description}

{f"ADDITIONAL REQUIREMENTS: {request.custom_requirements}" if request.custom_requirements else ""}

Generate the image now. Make it professional, clear, academically rigorous, and suitable for publication in academic materials."""
        
        return prompt
    
    def _validate_image_quality(self, response_text: str) -> dict:
        """Phase 3: Validate that response indicates successful image generation"""
        validation_data = {
            "has_image_generation": len(response_text) > 100,
            "quality_indicators": [],
            "needs_refinement": False,
            "confidence_score": 0.5
        }
        
        quality_keywords = [
            "professional", "clear", "readable", "hierarchy", "organized",
            "balanced", "appropriate", "academic", "publication", "quality",
            "structured", "systematic", "coherent", "logical"
        ]
        
        response_lower = response_text.lower()
        for keyword in quality_keywords:
            if keyword in response_lower:
                validation_data["quality_indicators"].append(keyword)
        
        if len(validation_data["quality_indicators"]) >= 3:
            validation_data["confidence_score"] = min(0.9, 0.5 + len(validation_data["quality_indicators"]) * 0.08)
        
        validation_data["needs_refinement"] = validation_data["confidence_score"] < 0.7
        
        return validation_data
    
    def _refine_request(self, request: ImageGenerationRequest, iteration: int, 
                       plan: dict, validation: dict) -> str:
        """Phase 4: Generate refinement prompt based on validation feedback"""
        refinement_prompt = f"""Refine the {request.image_type.value} for "{request.title}".

Iteration: {iteration}/3
Quality indicators found: {', '.join(validation.get('quality_indicators', []))}
Current confidence: {validation.get('confidence_score', 0.5):.0%}

Original design plan:
- Key elements: {', '.join(plan.get('key_elements', [])[:3])}
- Layout: {plan.get('layout_approach', 'best practices')}
- Color scheme: {plan.get('color_scheme', {}).get('primary', 'professional')}

Regenerate with emphasis on:
1. Crystal clear visual hierarchy and information flow
2. Academic professional styling and formatting
3. Consistent alignment and spacing
4. Maximum readability and clarity
5. Publication-ready appearance

Make this iteration significantly more polished. Focus on making the visualization immediately clear and professional to academic audience members.

Context: {request.description}"""
        
        return refinement_prompt
    
    def generate(self, request: ImageGenerationRequest) -> dict:
        """Main generation pipeline with agentic reasoning"""
        start_time = time.time()
        
        cache_key = self._get_cache_key(request)
        if cache_key in self.cache:
            return {
                "success": True,
                "image_data": self.cache[cache_key],
                "cached": True,
                "message": "Retrieved from cache"
            }
        
        metrics = {
            "planning_time": 0,
            "generation_time": 0,
            "validation_time": 0,
            "refinement_iterations": 0
        }
        
        try:
            # Phase 1: Planning
            plan_start = time.time()
            plan = self._plan_visualization(request)
            metrics["planning_time"] = time.time() - plan_start
            
            image_prompt = self._generate_image_prompt(request, plan)
            
            best_response = None
            best_validation = None
            
            # Phase 2-4: Generation with iterative refinement
            for iteration in range(request.max_retries):
                gen_start = time.time()
                
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=image_prompt
                )
                response_text = response.text
                metrics["generation_time"] += time.time() - gen_start
                
                # Phase 3: Validation
                val_start = time.time()
                validation = self._validate_image_quality(response_text)
                metrics["validation_time"] += time.time() - val_start
                
                best_response = response_text
                best_validation = validation
                
                # Check if quality is acceptable
                if not validation["needs_refinement"] or iteration == request.max_retries - 1:
                    break
                
                # Phase 4: Refinement
                metrics["refinement_iterations"] += 1
                refinement_prompt = self._refine_request(
                    request, iteration + 1, plan, validation
                )
                image_prompt = refinement_prompt
            
            total_time = time.time() - start_time
            
            # Cache the result
            self.cache[cache_key] = best_response
            
            return {
                "success": True,
                "image_data": best_response,
                "cached": False,
                "metadata": {
                    "request_title": request.title,
                    "image_type": request.image_type.value,
                    "complexity_level": request.complexity_level,
                    "visualization_plan": plan,
                    "validation_score": round(best_validation.get("confidence_score", 0), 3) if best_validation else 0,
                    "quality_indicators": best_validation.get("quality_indicators", []) if best_validation else [],
                    "generation_timestamp": datetime.now().isoformat(),
                    "metrics": {
                        "planning_time_seconds": round(metrics["planning_time"], 2),
                        "generation_time_seconds": round(metrics["generation_time"], 2),
                        "validation_time_seconds": round(metrics["validation_time"], 2),
                        "total_time_seconds": round(total_time, 2),
                        "refinement_iterations": metrics["refinement_iterations"]
                    }
                }
            }
        
        except ValueError as e:
            return {
                "success": False,
                "error": f"Validation Error: {str(e)}",
                "error_type": "validation_error"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error: {str(e)}",
                "error_type": "generation_error"
            }


class AcademicImageAPI:
    """Frontend-facing API wrapper"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.generator = AcademicImageGenerator(api_key)
    
    def create_flowchart(self, title: str, description: str, 
                        complexity_level: int = 2, 
                        custom_requirements: Optional[str] = None) -> dict:
        """Create a professional flowchart"""
        request = ImageGenerationRequest(
            title=title,
            description=description,
            image_type=ImageType.FLOWCHART,
            complexity_level=complexity_level,
            custom_requirements=custom_requirements,
            style_preferences={
                "shapes": "standard flowchart symbols",
                "connectors": "clear directional arrows",
                "decision_style": "diamond shapes"
            }
        )
        return self.generator.generate(request)
    
    def create_system_diagram(self, title: str, description: str,
                             complexity_level: int = 2,
                             custom_requirements: Optional[str] = None) -> dict:
        """Create a system architecture diagram"""
        request = ImageGenerationRequest(
            title=title,
            description=description,
            image_type=ImageType.SYSTEM_DIAGRAM,
            complexity_level=complexity_level,
            custom_requirements=custom_requirements,
            style_preferences={
                "layout": "hierarchical or layered",
                "components": "clearly distinguished boxes",
                "connections": "labeled relationships"
            }
        )
        return self.generator.generate(request)
    
    def create_process_flow(self, title: str, description: str,
                           complexity_level: int = 2,
                           custom_requirements: Optional[str] = None) -> dict:
        """Create a process flow diagram"""
        request = ImageGenerationRequest(
            title=title,
            description=description,
            image_type=ImageType.PROCESS_FLOW,
            complexity_level=complexity_level,
            custom_requirements=custom_requirements,
            style_preferences={
                "flow_direction": "left-to-right or top-to-bottom",
                "stages": "clearly marked phases",
                "transitions": "clear progression"
            }
        )
        return self.generator.generate(request)
    
    def create_conceptual_framework(self, title: str, description: str,
                                   complexity_level: int = 2,
                                   custom_requirements: Optional[str] = None) -> dict:
        """Create a conceptual framework diagram"""
        request = ImageGenerationRequest(
            title=title,
            description=description,
            image_type=ImageType.CONCEPTUAL_FRAMEWORK,
            complexity_level=complexity_level,
            custom_requirements=custom_requirements,
            style_preferences={
                "relationship_style": "connections showing relationships",
                "emphasis": "key concepts highlighted"
            }
        )
        return self.generator.generate(request)
    
    def create_data_visualization(self, title: str, description: str,
                                 complexity_level: int = 2,
                                 custom_requirements: Optional[str] = None) -> dict:
        """Create a data visualization"""
        request = ImageGenerationRequest(
            title=title,
            description=description,
            image_type=ImageType.DATA_VISUALIZATION,
            complexity_level=complexity_level,
            custom_requirements=custom_requirements,
            style_preferences={
                "chart_type": "appropriate to data",
                "labels": "axes and data point labels",
                "color_coding": "meaningful color representation"
            }
        )
        return self.generator.generate(request)
    
    def create_network_diagram(self, title: str, description: str,
                              complexity_level: int = 2,
                              custom_requirements: Optional[str] = None) -> dict:
        """Create a network diagram"""
        request = ImageGenerationRequest(
            title=title,
            description=description,
            image_type=ImageType.NETWORK_DIAGRAM,
            complexity_level=complexity_level,
            custom_requirements=custom_requirements,
            style_preferences={
                "node_style": "distinct and labeled",
                "connections": "clear relationship lines",
                "layout": "well-distributed spatial arrangement"
            }
        )
        return self.generator.generate(request)
    
    def create_timeline(self, title: str, description: str,
                       complexity_level: int = 2,
                       custom_requirements: Optional[str] = None) -> dict:
        """Create a timeline visualization"""
        request = ImageGenerationRequest(
            title=title,
            description=description,
            image_type=ImageType.TIMELINE,
            complexity_level=complexity_level,
            custom_requirements=custom_requirements,
            style_preferences={
                "flow_direction": "horizontal or vertical timeline",
                "time_markers": "clear date/period labels",
                "events": "distinct visual markers"
            }
        )
        return self.generator.generate(request)


if __name__ == "__main__":
    api = AcademicImageAPI()
    
    result = api.create_flowchart(
        title="Machine Learning Pipeline",
        description="A comprehensive ML pipeline showing data preprocessing, feature engineering, model training, and evaluation stages",
        complexity_level=3,
        custom_requirements="Include feedback loops and decision points for model selection"
    )
    
    if result["success"]:
        print("Generation successful!")
        print(f"Title: {result['metadata']['request_title']}")
        print(f"Type: {result['metadata']['image_type']}")
        print(f"Validation Score: {result['metadata']['validation_score']}")
        print(f"Total Time: {result['metadata']['metrics']['total_time_seconds']}s")
        print(f"\nImage Data:\n{result['image_data'][:500]}...")
    else:
        print(f"Generation failed: {result['error']}")