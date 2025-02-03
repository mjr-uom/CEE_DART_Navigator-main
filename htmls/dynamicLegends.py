def generate_legend_html(selected_labels):
    legend_items = {
        "gray": "exp",
        "red": "mut",
        "orange": "amp",
        "green": "del",
        "blue": "fus",
    }

    # Filter items based on selected labels
    filtered_items = {color: label for color, label in legend_items.items() if label in selected_labels}

    # Generate HTML for the legend
    legend_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Legend Table</title>
        <style>
                                            body {
                                               margin: 0; /* Remove default margin */
                                              padding: 10px; /* Add padding inside the border */
                                               border: 0px solid lightgray; /* Border around the entire page */
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
    """

    # Add selected legend items
    for color, label in filtered_items.items():
        legend_html += f"""
                <tr>
                    <td><span class="circle" style="background-color: {color};"></span></td>
                    <td>{label}</td>
                </tr>
        """

    legend_html += """
            </table>
        </div>
    </body>
    </html>
    """

    return legend_html
