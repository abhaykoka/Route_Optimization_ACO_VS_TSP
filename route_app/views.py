from django.shortcuts import render, redirect
from .forms import CitySelectionForm
from .Dataset_load import load_data
from .Optimal_TSP_Path import find_optimal_route
from django.shortcuts import render
from .Optimal_TSP_Graph import plot_exec_time
from .Ant_Colony_Optimization_Path import find_Optimal_path
from .Ant_Colony_Optimization_Graph import Aco_Graph
def input_view(request):
    file_path = 'route_app/static/Cities Dataset - Route Optimization.csv'
    _, _, cities = load_data(file_path)

    if request.method == 'POST':
        form = CitySelectionForm(request.POST)
        num_cities = int(request.POST.get('num_cities', 2))  # Retrieve selected number of cities
        form.generate_city_fields(cities, num_cities)
        
        if form.is_valid():
            selected_cities = [form.cleaned_data[f'city_{i}'] for i in range(num_cities)]
            request.session['selected_cities'] = selected_cities
            return redirect('output')
    else:
        form = CitySelectionForm()

    return render(request, 'input.html', {
        'form': form,
        'cities': cities
    })

def output_view(request):
    file_path = 'route_app/static/Cities Dataset - Route Optimization.csv'
    selected_cities = request.session.get('selected_cities')
    num_cities = len(selected_cities)
    # optimal_route, total_distance, execution_time = find_optimal_route(len(selected_cities), selected_cities, file_path)
    optimal_route, full_sequence, total_distance, execution_time = find_optimal_route(selected_cities)
    optimal_route_aco, full_sequence_aco, total_distance_aco, execution_time_aco = find_Optimal_path(selected_cities)
    plot_path = plot_exec_time(num_cities)
    aco_path = Aco_Graph(num_cities)
    return render(request, 'output.html', {
        'selected_cities' : ",".join(selected_cities),
        'optimal_route': optimal_route,
        'full_sequence': full_sequence,
        'total_distance': total_distance,
        'execution_time': execution_time,
        'plot_path' : plot_path,
        'optimal_route_aco': optimal_route_aco,
        'full_sequence_aco': full_sequence_aco,
        'total_distance_aco': total_distance_aco,
        'execution_time_aco': execution_time_aco,
        'Aco_path' : aco_path,
    })

def project_info(request):
    return render(request, 'project_info.html')
