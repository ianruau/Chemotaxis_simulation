#!/usr/bin/env python3
#
# By Le Chen and Chatgpt
# chenle02@gmail.com / le.chen@auburn.edu
# Created at Thu 03 Apr 2025 07:48:56 PM CDT
#

import os
import glob


def generate_html():
    # HTML template start
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chemotaxis Simulation Results</title>
        <style>
            .section {
                margin: 20px;
                padding: 20px;
                border: 1px solid #ccc;
            }
            .image-container {
                display: inline-block;
                margin: 10px;
            }
            img {
                max-width: 400px;
                height: auto;
            }
        </style>
    </head>
    <body>
    """

    # Process different chi values
    chi_values = [10, 23, 24, 70, 150, 300, 490, 700]

    for chi in chi_values:
        html_content += f'<div class="section"><h2>Chi = {chi}</h2>'

        # Find all images for this chi value
        pattern = f"*chi={chi}-0*.jpeg"
        images = glob.glob(pattern)
        images.sort()  # Sort images by name

        for image in images:
            html_content += f"""
            <div class="image-container">
                <img src="{image}" alt="{image}">
            </div>
            """

        html_content += '</div>'

    # Close HTML
    html_content += """
    </body>
    </html>
    """

    # Write to file
    with open('simulation_results.html', 'w') as f:
        f.write(html_content)


if __name__ == "__main__":
    generate_html()
