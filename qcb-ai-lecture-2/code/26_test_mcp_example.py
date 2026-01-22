import httpx
import xml.etree.ElementTree as ET
from fastmcp import FastMCP

mcp = FastMCP("PMC Fetching Test")

@mcp.tool
def get_abstract_by_pmcid(pmcid: str = None) -> str:
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmcid}"
    
    response = httpx.get(url, timeout=10.0)
    response.raise_for_status()
    
    root = ET.fromstring(response.content)
    abstract_elem = root.find(".//abstract")
    
    return "".join(abstract_elem.itertext())
    
if __name__ == "__main__":
    mcp.run()


