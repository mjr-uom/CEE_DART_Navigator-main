import matplotlib.colors as mcolors
import hashlib

def generate_legend_table(color_mapping):
    # Base HTML structure
    html_template = """<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Legend Table</title>
        <style>
            body {
                margin: 0px;
                padding: 5px;
                border: 0px solid lightgray;
                font-family: Arial, sans-serif;
            }
            .legend-title {
                font-size: 16px;
                font-weight: bold;
                margin-bottom: 10px;
                text-align: center;
                display: block;
                margin-left: auto;
                margin-right: auto;
            }
            .legend-table {
                width: auto;
                border-collapse: collapse;
                margin: 10px;
            }
            .legend-table td {
                padding: 8px 15px;
                font-family: Arial, sans-serif;
                font-size: 14px;
            }
            .circle {
                width: 15px;
                height: 15px;
                border-radius: 50%;
                display: inline-block;
            }
        </style>
    </head>
    <body>
        <div class="legend-container">
            <div class="legend-title">Nodes</div>
            <table class="legend-table">
                {table_rows}
            </table>
        </div>
    </body>
</html>"""

    # Generate table rows dynamically based on the input dictionary
    table_rows = ""
    for label, color in color_mapping.items():
        row = f"""
        <tr>
            <td><span class="circle" style="background-color: {color};"></span></td>
            <td>{label}</td>
        </tr>
        """
        table_rows += row.strip()

    # Insert rows into the HTML template
    html_output = html_template.replace("{table_rows}", table_rows)
    return html_output


def generate_legend_table_community(color_mapping):
    # Base HTML structure
    html_template = """<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Legend Table</title>
        <style>
            body {
                margin: 0px;
                padding: 5px;
                border: 0px solid lightgray;
                font-family: Arial, sans-serif;
            }
            .legend-title {
                font-size: 16px;
                font-weight: bold;
                margin-bottom: 10px;
                text-align: center;
                display: block;
                margin-left: auto;
                margin-right: auto;
            }
            .legend-table {
                width: auto;
                border-collapse: collapse;
                margin: 1px;
            }
            .legend-table td {
                padding: 8px 1px;
                font-family: Arial, sans-serif;
                font-size: 12px;
            }
            .circle {
                width: 15px;
                height: 15px;
                border-radius: 50%;
                display: inline-block;
            }
        </style>
    </head>
    <body>
        <div class="legend-container">
            <div class="legend-title">Communities</div>
            <table class="legend-table">
                {table_rows}
            </table>
        </div>
    </body>
</html>"""

    # Generate table rows dynamically based on the input dictionary
    table_rows = ""
    for label, color in color_mapping.items():
        row = f"""
        <tr>
            <td><span class="circle" style="background-color: {color};"></span></td>
            <td>Community {str(label)}</td>
        </tr>
        """
        table_rows += row.strip()

    # Insert rows into the HTML template
    html_output = html_template.replace("{table_rows}", table_rows)
    return html_output