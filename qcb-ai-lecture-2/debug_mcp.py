#!/usr/bin/env python3
"""
Debug script to test PMC API and understand response structure.
"""

import asyncio
import httpx
import xml.etree.ElementTree as ET


async def test_pmc_response():
    """Test different PMC IDs to find one with abstract"""
    
    # Try multiple PMC IDs
    pmcids = [
        "PMC7045128",  # COVID-19 paper
        "PMC6059045",  # Another recent paper
        "PMC5291611",  # Older paper
    ]
    
    for pmcid in pmcids:
        print(f"\nTesting {pmcid}...")
        try:
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmcid}"
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                
            # Parse XML
            root = ET.fromstring(response.content)
            
            # Print first few hundred chars to see structure
            xml_str = ET.tostring(root, encoding='unicode')
            print(f"  XML length: {len(xml_str)} chars")
            
            # Try different XPath expressions
            abstract_elem = root.find(".//abstract")
            if abstract_elem is not None:
                text = "".join(abstract_elem.itertext())
                text = " ".join(text.split())
                print(f"  ✓ Found abstract with {len(text)} characters")
                print(f"    Preview: {text[:150]}...")
                return pmcid
            else:
                print(f"  ✗ No abstract element found")
                
                # Check what elements are present
                sections = root.findall(".//sec")
                print(f"    Sections found: {len(sections)}")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\nNo suitable article found, but API is working")
    return None


async def test_simple_case():
    """Test with a completely minimal example"""
    print("\nTesting with very simple fetch...")
    
    # This one is known to have a valid XML structure
    pmcid = "PMC3879496"
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmcid}"
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            print(f"✓ API response received, status: {response.status_code}")
            print(f"  Response size: {len(response.content)} bytes")
            
            # Try to parse
            try:
                root = ET.fromstring(response.content)
                print(f"✓ XML parsed successfully")
                
                # Show available elements
                articles = root.findall(".//article")
                print(f"  Articles in response: {len(articles)}")
                
                if articles:
                    art = articles[0]
                    # List immediate children
                    for child in list(art)[:5]:
                        print(f"    - {child.tag}")
                
                return True
            except ET.ParseError as e:
                print(f"✗ XML parse error: {e}")
                return False
                
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


async def main():
    print("=" * 60)
    print("Debugging PMC Fetcher")
    print("=" * 60)
    
    await test_simple_case()
    # await test_pmc_response()


if __name__ == "__main__":
    asyncio.run(main())
