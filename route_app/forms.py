# route_app/forms.py

from django import forms

class CitySelectionForm(forms.Form):
    num_cities = forms.IntegerField(
        label="Number of Cities",
        min_value=2,
        max_value=44,
        widget=forms.NumberInput(attrs={'id': 'num-cities-input'})
    )

    def generate_city_fields(self, cities, num_cities):
        for i in range(num_cities):
            self.fields[f'city_{i}'] = forms.ChoiceField(
                choices=[(city, city) for city in cities],
                label=f'Select City {i + 1}',
                widget=forms.Select(attrs={'class': 'city-dropdown'})
            )
