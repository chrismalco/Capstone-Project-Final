import streamlit as st

# Define the list of sources with URLs and descriptions
sources = [
    {
        "name": "GENERAL SPECIFICATION APPENDIX A SAFETY, HEALTH AND ENVIRONMENT (April 2024 Edition)",
        "url": "https://file.go.gov.sg/gsappa042024.pdf",
        "description": (
            "The Land Transport Authority (LTA) General Specification Appendix A - Safety, Health, and Environment (April 2024 Edition) likely serves as a regulatory document detailing "
            "standards, protocols, and requirements for ensuring safe, healthy, and environmentally responsible practices within a particular project, "
            "industry, or organization. This appendix typically outlines guidelines to:\n\n"
            "- **Promote workplace safety** by minimizing hazards and managing risks.\n"
            "- **Uphold health standards** to protect the well-being of all personnel.\n"
            "- **Support environmental protection** through sustainable practices and pollution prevention.\n\n"
            "It may include specific safety procedures, compliance obligations, emergency response protocols, and environmental management systems that align "
            "with the latest regulations as of April 2024."
        )
    },
]

# Set the page title
st.title("List of Sources")

# Display each source with a link and description
for source in sources:
    st.subheader(source["name"])
    st.write(source["description"])
    st.markdown(f"[Visit Website]({source['url']})", unsafe_allow_html=True)