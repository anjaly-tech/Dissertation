import itertools
import sympy as sp
import random
import re
import matplotlib.pyplot as plt
from sympy import Rational
import numpy as np
import pandas as pd
import seaborn as sns
from texttable import Texttable

def get_random_outputs(num_of_variables):
    """Generate all combinations of binary variables and assign random outputs."""
    #print("\nGenerating all combinations of binary variables:")
    output_values = []
    # Generate all binary combinations for the given number of variables.
    for combination in itertools.product([0, 1], repeat=num_of_variables):
        output = random.choice([0, 1])  # Randomly assign an output of 0 or 1.
        #print(f"  Combination: {combination} -> Output: {output}")
        output_values.append(output)  # Store the output for each combination.
    return output_values

def generate_sop_expression(variables, truth_values):
    """Generate the Sum of Products (SOP) boolean expression."""
    all_combinations = list(itertools.product([0, 1], repeat=len(variables)))
    terms = []
    # Create a term for each combination where the output is 1.
    for combination, value in zip(all_combinations, truth_values):
        if value:
            term = []
            # Create a sub-term for each variable in the combination.
            for var, val in zip(variables, combination):
                term.append(f"({var})" if val else f"(¬{var})")
            terms.append(" AND ".join(term))  # Join the sub-terms with AND.
    # Join the terms with OR to get the final SOP expression.
    sop_expression = " OR ".join(f"({term})" for term in terms) if terms else "False"
    return sop_expression

def get_fourier(boolean_output, num_of_variables):
    """Compute the fourier_expression for the boolean function."""
    #print("\nComputing the simplify_expression for the boolean function:")
    base = [[1, -1] for _ in range(num_of_variables)]  # Define the base for the simplify_expression.
    terms = []

    # Generate all combinations of the base and compute the simplify_expression term for each combination.
    for idx, base_vals in enumerate(itertools.product(*base)):
        term_list = [f"((1 + {base_val}*x{i+1})/2)" for i, base_val in enumerate(base_vals)]
        term_list.append(str(boolean_output[idx]))
        mul_str = "*".join(term_list)
        terms.append(mul_str)

    # Sort the terms based on the number of variables and constant term.
    terms.sort(key=lambda term: (term.count('x'), term))

    simplify_expression_string = " + ".join(terms)
    simplify_expression = sp.simplify(simplify_expression_string)  # Simplify the simplify_expression.
    #print("  ", simplify_expression)
    return simplify_expression



def extract_terms(poly):
    """Extract terms from a simplify_expression."""
    # Split the simplify_expression into terms.
    terms = re.split(r'\s*\+\s*', poly.replace('-', '+-')) 
    # Clean the terms by removing any leading or trailing spaces.
    terms = [term.strip() for term in terms if term.strip()]
    # Sort the terms based on the number of variables in each term.
    return sorted(terms, key=lambda x: len(re.findall(r'x', x)))

def display_monomials(poly):
    """Display simplify_expression terms in a sorted order."""
    #print("\nThe sorted simplify_expression representation of the Boolean function 1 is:")
    terms = extract_terms(poly)  # Extract the terms from the simplify_expression.
    # Handle the constant term separately.
    constant = [term for term in terms if 'x' not in term]
    if constant:
        print(f"  Constant term: {constant[0]}")
        terms.remove(constant[0])  # Remove the constant term from the terms list.
    curr_num_vars = 1
    # Loop through sorted terms and print based on the number of variables.
    while terms:
        monomials = [term for term in terms if len(re.findall(r'x', term)) == curr_num_vars]
        if monomials:
            print(f"  Monomials with {curr_num_vars} variable{'s' if curr_num_vars>1 else ''}:")
            for monomial in monomials:
                print("    ", monomial)
                terms.remove(monomial)  # Remove the displayed term from the terms list.
        curr_num_vars += 1

def get_coefficients(poly):
    """Get coefficients of each monomial in a simplify_expression."""
    monomials = []
    coefficients = []
    # Extract the coefficient and monomial from each term in the simplify_expression.
    for term in poly.as_ordered_terms():
        coefficient, monomial = term.as_coeff_mul()
        monomial = '*'.join(map(str, monomial))
        if not monomial:  # If there's no variable, set monomial to '1' for the constant term.
            monomial = '1'
        monomials.append(monomial)
        coefficients.append(float(Rational(coefficient)))  # Convert coefficient to a floating point.
    return monomials, coefficients

def plot_graph(monomials, coefficients):
    """Plot a graph of the simplify_expression coefficients."""
    plt.figure(figsize=(10, 5))
    abbreviated_monomials = [f'M{i+1}' for i, _ in enumerate(monomials)]  # Abbreviate the x-axis labels

    # Use the first color from seaborn's deep palette
    color = sns.color_palette("deep")[0]

    # Same color, grid, and styles for consistency
    plt.plot(abbreviated_monomials, coefficients, color= color, marker='o', linewidth=2.5, markersize=10)
    plt.grid(True)

    # Set the font size and weight for the labels and title
    plt.xlabel('Monomials', fontsize=14)
    plt.ylabel('Coefficients', fontsize=14)
    plt.title('Graph of the simplify_expression Coefficients', fontsize=16)
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

    print("Monomial Legend:")
    for i, monomial in enumerate(monomials):
        print(f'M{i+1}: {monomial}')

def process_multiple_functions(num_functions, num_of_variables):
    results = []

    # Loop through and generate and analyze each function.
    for i in range(num_functions):
        print("\n" + "*" * 60)  # Separator
        print(f" Displaying details for Function {i+1}:")
        combinations = get_random_outputs(num_of_variables)  # Get the random outputs for each combination.

        variables = [f'x{i+1}' for i in range(num_of_variables)]  # Generate variable names.
        expr = generate_sop_expression(variables, combinations)  # Generate SOP expression.
        #print(f"\n  Expression:\n    {expr}")  # Display the SOP expression.

        print("\n  Truth table:")
        t = Texttable()
        t.header(['Inputs', 'Output'])
        all_combinations = list(itertools.product([0, 1], repeat=len(variables)))
        for combination, value in zip(all_combinations, combinations):
            t.add_row([str(combination), value])
        print(t.draw())

        output = [1 if value == 0 else -1 for value in combinations]  # Map output values to 1 or -1 in order to generate the fourier expression for the corresponding boolean function.
        simplify_expression = get_fourier(output, num_of_variables)  # Compute the simplify_expression for the boolean function.

        print("\n  Fourier Expression: ")
        print(f"    {simplify_expression}")

        monomials, coefficients = get_coefficients(simplify_expression)  # Get the coefficients of each monomial.
        sum_abs_coefficients = sum(abs(coeff) for coeff in coefficients)  # Sum the absolute values of the coefficients.

        print("\n Sum of Absolute Values: ")
        print(f" {sum(abs(coeff) for coeff in coefficients) } ")

        results.append({
            "Function": f"f{i+1}",
            "Expression": expr,
            "simplify_expression": simplify_expression,
            "Sum of Absolute Values of Coefficients": sum_abs_coefficients
        })

    return pd.DataFrame(results)

def main():
    width = 100  # Width of the output line
    text1 = "Boolean Function and Fourier Expression Generator"
    print("-" * width)
    print(text1.center(width))
    print("-" * width)
    
    num_of_variables = int(input("Please enter the number of variables: "))
    num_functions =   int(input("Please enter the number of random functions to be generated: "))

    text2= "The following output presents the Evaluation and Fourier simplify_expression representations of random Boolean functions with the number of variables that user inputs. Each section provides the truth table, and simplify_expression representation and the sum of absolute values of the respective Boolean function."
    print("-" * width)
    print(text2.center(width))
    print("-" * width)

    results_df = process_multiple_functions(num_functions, num_of_variables)

    # Display only the sum of absolute values of coefficients for each function
    sum_abs_values_df = results_df[['Function', 'Sum of Absolute Values of Coefficients']]
    print("-" * width)
    print("\nThis table provides a summary of the sum of the absolute values of coefficients for each generated function:")
    print(sum_abs_values_df)

    print("\nThe graph below visualizes the sum of the absolute values of the simplify_expression coefficients for each function:\n")


    # Plot the graph for sum of absolute values of coefficients vs function number
    plt.figure(figsize=(10, 5))

    # Use the first color from seaborn's deep palette
    color = sns.color_palette("deep")[0]
    
    # Same color, grid, and styles for consistency
    sns.lineplot(data=results_df, x='Function', y='Sum of Absolute Values of Coefficients', color= color, marker='o')
    plt.grid(True)

    # Set the font size and weight for the labels and title
    plt.xlabel('Function Number', fontsize=14)
    plt.ylabel('Sum of Absolute Values of Coefficients', fontsize=14)
    plt.title('Sum of Absolute Values of Coefficients for Each Function', fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

    print("\nDetailed analysis of the first generated function:\n")


     # Display the detailed results for the first function
    display_monomials(str(results_df.iloc[0]['simplify_expression']))  # Display each simplify_expression term for the corresponding monomials.

    

    monomials, coefficients = get_coefficients(results_df.iloc[0]['simplify_expression'])  # Get the coefficients of each monomial.
    
    print("\nThe monomials are represented on the graph by the following labels:\n")


    plot_graph(monomials, coefficients)  # Plot the coefficients.

if __name__ == "__main__":
    main()


