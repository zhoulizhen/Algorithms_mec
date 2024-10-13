import folium

def constructMap(cls,ues):
    map = folium.Map(location=[-37.81517, 144.97476], zoom_start=12)

    # Add cloudlets to map
    for i in range(len(cls)):
        tooltip = 'Hello'
        folium.Marker([cls['LATITUDE'][i],cls['LONGITUDE'][i]], popup=('10003027'), tooltip=tooltip, icon=folium.Icon(color='green')).add_to(map)
        folium.Circle(
            location=[cls['LATITUDE'][i],cls['LONGITUDE'][i]],
            radius=200,
            color='#3186cc',
            fill=True,
            fill_color='#3186cc',
            fill_opacity=0.3,
        ).add_to(map)

    # Add user requests to map
    for i in range(len(ues)):
        tooltip = 'Hello'
        folium.Marker([ues['Latitude'][i],ues['Longitude'][i]], popup=('10003027'), tooltip=tooltip, icon=folium.Icon(color='orange', icon=None)).add_to(map)

    map.save('map.html')
