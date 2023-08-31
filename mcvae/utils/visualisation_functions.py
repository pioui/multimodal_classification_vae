def generate_latex_confusion_matrix(matrix, caption=" "):

    matrix = matrix[1:,1:]
    n = len(matrix)
    
    # Begin LaTeX table
    latex_code = "\\begin{table}[ht]\n"
    latex_code += "\\centering\n"
    latex_code += "\\caption{ Confusion matrix of: "+caption+"}\n"
    latex_code += "\\label{}\n"

    # Begin tabular environment
    latex_code += "\\begin{tabular}{c" + "c" * n + "}\n"
    latex_code += "\\toprule\n"
    
    # Column headers
    latex_code += " & " + " & ".join(["$ c_{"+str(i+1)+"} $" for i in range(n)]) + " \\\\\n"
    latex_code += "\\midrule\n"
    
    # Matrix values
    for i in range(n):
        latex_code += "$ c_{"+str(i+1)+ "}$ & " + " & ".join([str(matrix[i][j]) for j in range(n)]) + " \\\\\n"
    latex_code += "\\bottomrule\n"
    
    # End tabular environment
    latex_code += "\\end{tabular}\n"
    
    # End table environment
    latex_code += "\\end{table}"
    
    return latex_code

def generate_latex_matrix_from_dict(indict, caption=" "):

    latex_code = ''
    for row_name in indict.keys():
        latex_code += row_name
        latex_code += " & " + " & ".join([str(round(i,2)) for i in indict[row_name]]) + " \\\\\n"
    
    return latex_code