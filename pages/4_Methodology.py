import streamlit as st
import graphviz

def create_flowchart():
    # Create a flowchart using Graphviz
    dot = graphviz.Digraph()

    # Define nodes and edges based on the specified steps
    dot.node('A', 'Start')
    dot.node('B', 'User clicks the use cases they are interested in:\na) Safety\nb) Health\nc) Environment')
    dot.node('C', 'Scrape source of data and online PDF file')
    dot.node('D', 'LLM filters information based on selected use case and query')
    dot.node('E', 'Response relevant to chosen topic plus the query')
    dot.node('F', 'No relevant information found?')
    dot.node('G', 'End')

    # Define edges
    dot.edges(['AB', 'BC', 'CD', 'DE'])
    dot.edge('E', 'G', label='Yes')
    dot.edge('F', 'G', label='No', constraint='false')

    return dot

def main():
    st.title("Flowchart Creation")

    # Create the flowchart
    flowchart = create_flowchart()

    # Render the flowchart in the Streamlit app
    st.subheader("Process Flowchart")
    st.graphviz_chart(flowchart)

    st.write("""
    This flowchart illustrates the process flow for the chatbot application:
    1. Start
    2. User selects a use case (Safety, Health, Environment)
    3. Application scrapes data from the specified PDF
    4. LLM filters information based on the user's query
    5. Response is given based on the topic or indicates if no relevant information is found
    6. End
    """)

if __name__ == "__main__":
    main()