import kivy 
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.pagelayout import PageLayout
from kivy.uix.anchorlayout import AnchorLayout

class Index(Widget):
    pass 
        
class Analise(AnchorLayout):
    pass

class FamachApp(App):
    def build(self):
        self.title = "FAMACHAPP"
        self.icon = "../Imagens/logo.png"
        return Index()

if __name__ == "__main__":
    FamachApp().run()