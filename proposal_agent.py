"""
Proposal Agent - Gathers detailed business information and creates proposals
"""
import logging
import re
from typing import Dict, List, Optional
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
import markdown2  # For converting markdown to HTML

logger = logging.getLogger(__name__)

class ProposalAgent:
    def __init__(self):
        self.deep_questions = [
            "What is the current size of your team and organizational structure?",
            "What are your main revenue streams and how do they contribute to your overall business?",
            "What specific metrics or KPIs do you currently track for your business performance?",
            "What technology stack or tools are you currently using in your operations?",
            "Who are your main competitors and how do you differentiate from them?",
            "What are your short-term (6 months) and long-term (1-2 years) business goals?",
            "What is your current customer acquisition strategy and what channels work best?",
            "What operational challenges are consuming most of your time and resources?",
            "What financial constraints or budget limitations do you face?",
            "What regulatory or compliance requirements affect your industry?"
        ]
        self.current_question_index = 0
        self.gathered_info = {}
        self.company_context = {}
        
        # Initialize custom styles
        self.styles = self._create_custom_styles()

    def _create_custom_styles(self) -> Dict[str, ParagraphStyle]:
        """Create custom styles for the PDF document"""
        styles = getSampleStyleSheet()
        
        custom_styles = {}
        
        # Title style
        custom_styles['Title'] = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1,  # Center
            textColor=HexColor('#2c3e50'),  # Dark blue
            fontName='Helvetica-Bold'
        )
        
        # Heading style
        custom_styles['Heading'] = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=10,
            textColor=HexColor('#34495e'),  # Slightly lighter blue
            fontName='Helvetica-Bold'
        )
        
        # Normal text style
        custom_styles['Normal'] = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=11,
            spaceBefore=6,
            spaceAfter=6,
            leading=14,  # Line spacing
            fontName='Helvetica'
        )
        
        return custom_styles

    def sanitize_text(self, text: str) -> str:
        """Sanitize text for PDF generation"""
        if not text:
            return "N/A"
        # Remove any non-printable characters
        text = ''.join(char for char in text if char.isprintable())
        # Convert to ASCII to handle any encoding issues
        text = text.encode('ascii', 'ignore').decode('ascii')
        # Remove excessive whitespace
        text = ' '.join(text.split())
        return text.strip()

    def markdown_to_reportlab(self, markdown_text: str) -> List[Paragraph]:
        """Convert markdown text to a list of ReportLab paragraphs with robust error handling"""
        if not markdown_text or not isinstance(markdown_text, str):
            return [Paragraph("No content available.", self.styles['Normal'])]
        
        try:
            # Convert markdown to HTML
            html = markdown2.markdown(markdown_text)
            
            if not html or not isinstance(html, str):
                # Fallback to plain text
                return [Paragraph(self.sanitize_text(markdown_text), self.styles['Normal'])]
            
            # Split into sections based on headers
            sections = []
            current_section = []
            
            for line in html.split('\n'):
                line = line.strip()
                if line:
                    if line.startswith('<h1>'):
                        if current_section:
                            sections.append(current_section)
                        current_section = [line]
                    else:
                        current_section.append(line)
            
            if current_section:
                sections.append(current_section)
            
            # If no sections found, treat entire content as one section
            if not sections:
                sections = [[html]]
            
            # Convert sections to ReportLab paragraphs
            paragraphs = []
            for section in sections:
                for line in section:
                    if not line or not isinstance(line, str):
                        continue
                    
                    try:
                        style = None
                        clean_line = line
                        
                        # Extract text and determine style
                        if '<h1>' in line:
                            style = self.styles['Title']
                            clean_line = re.sub(r'<h1>(.*?)</h1>', r'\1', line, flags=re.DOTALL)
                            clean_line = re.sub(r'<[^>]+>', '', clean_line)  # Remove any remaining tags
                        elif '<h2>' in line:
                            style = self.styles['Heading']
                            clean_line = re.sub(r'<h2>(.*?)</h2>', r'\1', line, flags=re.DOTALL)
                            clean_line = re.sub(r'<[^>]+>', '', clean_line)
                        elif '<h3>' in line:
                            style = self.styles['Heading']
                            clean_line = re.sub(r'<h3>(.*?)</h3>', r'\1', line, flags=re.DOTALL)
                            clean_line = re.sub(r'<[^>]+>', '', clean_line)
                        else:
                            style = self.styles['Normal']
                            # Clean HTML tags but preserve basic formatting
                            clean_line = line
                        
                        clean_line = self.sanitize_text(clean_line)
                        if clean_line and clean_line.strip():
                            # Use clean_line directly if it's HTML, otherwise escape it
                            try:
                                paragraphs.append(Paragraph(clean_line, style))
                                paragraphs.append(Spacer(1, 6))
                            except Exception as para_error:
                                # If paragraph creation fails, try with escaped HTML
                                logger.warning(f"Paragraph creation failed: {str(para_error)}, using plain text")
                                escaped_line = clean_line.replace('<', '&lt;').replace('>', '&gt;')
                                paragraphs.append(Paragraph(escaped_line, style))
                                paragraphs.append(Spacer(1, 6))
                    except Exception as line_error:
                        logger.warning(f"Error processing line: {str(line_error)}, skipping")
                        continue
            
            # Ensure we return at least one paragraph
            if not paragraphs:
                return [Paragraph(self.sanitize_text(markdown_text), self.styles['Normal'])]
            
            return paragraphs
        except Exception as e:
            logger.error(f"Error converting markdown to reportlab: {e}")
            # Return sanitized text as normal paragraph if conversion fails
            try:
                sanitized = self.sanitize_text(markdown_text)
                return [Paragraph(sanitized if sanitized else "Content unavailable.", self.styles['Normal'])]
            except Exception as fallback_error:
                logger.error(f"Even fallback paragraph creation failed: {str(fallback_error)}")
                # Last resort: return empty list, caller should handle this
                return [Paragraph("Content could not be processed.", self.styles['Normal'])]

    def should_ask_deep_questions(self, user_info: Dict) -> bool:
        """Determine if we should ask deep questions based on conversation length"""
        # This will be called by main chat to decide when to switch to deep questioning
        return True

    def get_next_deep_question(self, user_info: Dict) -> str:
        """Get the next deep question to ask"""
        if self.current_question_index < len(self.deep_questions):
            question = self.deep_questions[self.current_question_index]
            self.current_question_index += 1
            return f"<h3>Let's dive deeper into your business</h3><p>{question}</p>"
        else:
            return self.propose_proposal_generation(user_info)

    def propose_proposal_generation(self, user_info: Dict) -> str:
        """Ask user if they want to generate a proposal"""
        return """<h3>Ready for Your Business Proposal</h3>
        <p>Based on our detailed discussion, I have gathered comprehensive information about your business. 
        I can now create a detailed problem proposal document that will help our planner agent develop 
        a strategic solution for your challenges.</p>
        <p><strong>Would you like me to generate a detailed proposal document?</strong></p>
        <p>Type 'yes' to proceed with proposal generation, or continue our conversation if you'd like to provide more details.</p>"""

    def generate_proposal_pdf(self, user_info: Dict, conversation_history: List[str], output_path: str) -> str:
        """Generate a detailed PDF proposal document - handles all edge cases to ensure PDF is always generated"""
        import os
        import shutil
        
        # Validate and normalize inputs with fallbacks
        if not user_info or not isinstance(user_info, dict):
            logger.warning("Invalid user_info provided, using empty dict")
            user_info = {}
        
        if not conversation_history or not isinstance(conversation_history, list):
            logger.warning("Invalid conversation_history provided, using empty list")
            conversation_history = []
        
        if not output_path or not isinstance(output_path, str):
            logger.error("Invalid output_path provided")
            raise ValueError("output_path must be a valid string")

        # Ensure directory exists with error handling
        try:
            output_dir = os.path.dirname(output_path)
            if output_dir:  # Only create if there's a directory path
                os.makedirs(output_dir, exist_ok=True)
        except (OSError, PermissionError) as e:
            logger.error(f"Failed to create output directory: {str(e)}")
            # Try to use current directory as fallback
            output_path = os.path.join(os.getcwd(), os.path.basename(output_path))
            logger.info(f"Using fallback path: {output_path}")

        # Ensure we have at least some content to generate
        if not conversation_history and not user_info:
            # Create minimal content to ensure PDF generation
            conversation_history = ["Business problem proposal document."]

        try:
            # Initialize PDF document
            doc = SimpleDocTemplate(output_path, pagesize=letter)
            story = []

            # Title
            story.append(Paragraph("Business Problem Analysis", self.styles['Title']))
            story.append(Spacer(1, 30))

            # Company Information - handle missing fields gracefully
            story.append(Paragraph("Company Information", self.styles['Heading']))
            company_fields = [
                ('company_name', 'Company Name'),
                ('industry', 'Industry'),
                ('company_size', 'Company Size'),
                ('revenue', 'Revenue Range'),
                ('role', 'Primary Role'),
                ('country', 'Country')
            ]
            
            has_company_info = False
            for field, label in company_fields:
                value = user_info.get(field, '')
                if value:  # Only add non-empty fields
                    safe_value = self.sanitize_text(str(value))
                    story.append(Paragraph(f"<b>{label}:</b> {safe_value}", self.styles['Normal']))
                    has_company_info = True
            
            if not has_company_info:
                story.append(Paragraph("Company information not provided.", self.styles['Normal']))
            
            story.append(Spacer(1, 20))

            # Business Challenge - always include this section
            story.append(Paragraph("Business Challenge", self.styles['Heading']))
            challenge = user_info.get('challenge', 'Not specified')
            if not challenge or challenge == 'Not specified':
                challenge = "Business challenge details to be provided."
            challenge = self.sanitize_text(str(challenge))
            story.append(Paragraph(challenge, self.styles['Normal']))
            story.append(Spacer(1, 20))

            # Process markdown content from conversation history - handle errors gracefully
            # The generated content will have its own section headings (Executive Summary, Problem Statement, etc.)
            if conversation_history:
                story.append(Spacer(1, 10))
                
                for content in conversation_history:
                    if content and isinstance(content, str):
                        try:
                            # Convert markdown content to ReportLab paragraphs
                            paragraphs = self.markdown_to_reportlab(content)
                            if paragraphs:
                                story.extend(paragraphs)
                                story.append(Spacer(1, 10))
                        except Exception as e:
                            logger.warning(f"Error processing markdown content: {str(e)}, using plain text")
                            # Fallback to plain text
                            safe_content = self.sanitize_text(content)
                            if safe_content:
                                story.append(Paragraph(safe_content, self.styles['Normal']))
                                story.append(Spacer(1, 10))
            else:
                # Ensure we have content even if conversation_history is empty
                story.append(Paragraph("Executive Summary", self.styles['Heading']))
                story.append(Paragraph("Business problem analysis based on conversation.", self.styles['Normal']))
                story.append(Spacer(1, 20))

            # Generated Date/Time - Footer (as per requirements)
            story.append(PageBreak())
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            story.append(Paragraph("Generated Date/Time", self.styles['Heading']))
            story.append(Paragraph(f"{timestamp}", self.styles['Normal']))

            # Build PDF with retry logic
            try:
                doc.build(story)
                logger.info(f"Proposal PDF generated successfully at {output_path}")
                # Verify file was created
                if not os.path.exists(output_path):
                    raise Exception("PDF file was not created")
                return output_path
            except Exception as build_error:
                logger.error(f"Error building PDF: {str(build_error)}")
                # Try with minimal content as last resort
                logger.info("Attempting to generate PDF with minimal content as fallback")
                try:
                    fallback_doc = SimpleDocTemplate(output_path, pagesize=letter)
                    fallback_story = [
                        Paragraph("Business Problem Analysis", self.styles['Title']),
                        Spacer(1, 30),
                        Paragraph("Business Challenge", self.styles['Heading']),
                        Paragraph(self.sanitize_text(str(user_info.get('challenge', 'Business problem to be addressed'))), self.styles['Normal']),
                        Spacer(1, 20),
                        Paragraph(f"Generated on: {timestamp}", self.styles['Normal'])
                    ]
                    fallback_doc.build(fallback_story)
                    logger.info(f"Proposal PDF generated with fallback content at {output_path}")
                    return output_path
                except Exception as fallback_error:
                    logger.error(f"Fallback PDF generation also failed: {str(fallback_error)}")
                    raise Exception(f"Failed to generate proposal PDF after multiple attempts: {str(fallback_error)}")

        except Exception as e:
            logger.error(f"Unexpected error in generate_proposal_pdf: {str(e)}")
            # Last resort: try to create a minimal PDF
            try:
                minimal_path = output_path.replace('.pdf', '_minimal.pdf') if '.pdf' in output_path else output_path + '_minimal.pdf'
                minimal_doc = SimpleDocTemplate(minimal_path, pagesize=letter)
                minimal_story = [
                    Paragraph("Business Problem Analysis", self.styles['Title']),
                    Spacer(1, 30),
                    Paragraph("This problem analysis document was generated with minimal content due to an error.", self.styles['Normal']),
                    Spacer(1, 20),
                    Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", self.styles['Normal'])
                ]
                minimal_doc.build(minimal_story)
                logger.info(f"Minimal PDF generated as last resort at {minimal_path}")
                return minimal_path
            except Exception as final_error:
                logger.error(f"Even minimal PDF generation failed: {str(final_error)}")
                raise Exception(f"Failed to generate proposal PDF: {str(final_error)}")

    def store_gathered_info(self, key: str, value: str):
        """Store information gathered during deep questioning"""
        self.gathered_info[key] = value

    def reset_agent(self):
        """Reset the agent for a new conversation"""
        self.current_question_index = 0
        self.gathered_info = {}
        self.company_context = {}

    def generate_solution_pdf(self, markdown_report: str, output_path: str) -> str:
        """Generate a solution report PDF from markdown content
        
        Args:
            markdown_report: Markdown formatted solution report
            output_path: Path where PDF will be saved
            
        Returns:
            Path to the generated PDF file
        """
        import os
        
        if not markdown_report or not isinstance(markdown_report, str):
            logger.warning("Invalid markdown_report provided, using minimal content")
            markdown_report = "Solution Report - Content unavailable"
        
        if not output_path or not isinstance(output_path, str):
            logger.error("Invalid output_path provided")
            raise ValueError("output_path must be a valid string")

        # Ensure directory exists
        try:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
        except (OSError, PermissionError) as e:
            logger.error(f"Failed to create output directory: {str(e)}")
            output_path = os.path.join(os.getcwd(), os.path.basename(output_path))

        try:
            # Initialize PDF document
            doc = SimpleDocTemplate(output_path, pagesize=letter)
            story = []

            # Add title
            story.append(Paragraph("Comprehensive Solution Report", self.styles['Title']))
            story.append(Spacer(1, 30))

            # Convert markdown to ReportLab paragraphs
            paragraphs = self.markdown_to_reportlab(markdown_report)
            if paragraphs:
                story.extend(paragraphs)
            else:
                # Fallback to plain text
                story.append(Paragraph(self.sanitize_text(markdown_report), self.styles['Normal']))

            # Add footer
            story.append(PageBreak())
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            story.append(Paragraph("Generated Date/Time", self.styles['Heading']))
            story.append(Paragraph(f"{timestamp}", self.styles['Normal']))

            # Build PDF
            try:
                doc.build(story)
                logger.info(f"Solution PDF generated successfully at {output_path}")
                if not os.path.exists(output_path):
                    raise Exception("PDF file was not created")
                return output_path
            except Exception as build_error:
                logger.error(f"Error building solution PDF: {str(build_error)}")
                # Fallback: try with minimal content
                fallback_doc = SimpleDocTemplate(output_path, pagesize=letter)
                fallback_story = [
                    Paragraph("Comprehensive Solution Report", self.styles['Title']),
                    Spacer(1, 30),
                    Paragraph(self.sanitize_text(markdown_report[:1000]), self.styles['Normal']),
                    Spacer(1, 20),
                    Paragraph(f"Generated on: {timestamp}", self.styles['Normal'])
                ]
                fallback_doc.build(fallback_story)
                logger.info(f"Solution PDF generated with fallback at {output_path}")
                return output_path

        except Exception as e:
            logger.error(f"Unexpected error in generate_solution_pdf: {str(e)}")
            # Last resort: create minimal PDF
            try:
                minimal_path = output_path.replace('.pdf', '_minimal.pdf') if '.pdf' in output_path else output_path + '_minimal.pdf'
                minimal_doc = SimpleDocTemplate(minimal_path, pagesize=letter)
                minimal_story = [
                    Paragraph("Comprehensive Solution Report", self.styles['Title']),
                    Spacer(1, 30),
                    Paragraph("Solution report generated with minimal content due to an error.", self.styles['Normal']),
                    Spacer(1, 20),
                    Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", self.styles['Normal'])
                ]
                minimal_doc.build(minimal_story)
                logger.info(f"Minimal solution PDF generated at {minimal_path}")
                return minimal_path
            except Exception as final_error:
                logger.error(f"Even minimal solution PDF generation failed: {str(final_error)}")
                raise Exception(f"Failed to generate solution PDF: {str(final_error)}")


