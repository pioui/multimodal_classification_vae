def generate_latex_confusion_matrix(matrix):
    n = len(matrix)
    
    # Begin LaTeX table
    latex_code = "\\begin{table}[ht]\n"
    latex_code += "\\centering\n"
    latex_code += "\\caption{}\n"
    latex_code += "\\label{}\n"

    # Begin tabular environment
    latex_code += "\\begin{tabular}{c" + "c" * n + "}\n"
    latex_code += "\\toprule\n"
    
    # Column headers
    latex_code += " & " + " & ".join([str(i) for i in range(n)]) + " \\\\\n"
    latex_code += "\\midrule\n"
    
    # Matrix values
    for i in range(n):
        latex_code += str(i) + " & " + " & ".join([str(matrix[i][j]) for j in range(n)]) + " \\\\\n"
    latex_code += "\\bottomrule\n"
    
    # End tabular environment
    latex_code += "\\end{tabular}\n"
    
    # End table environment
    latex_code += "\\end{table}"
    
    return latex_code