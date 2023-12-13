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
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup
from kivy.core.window import Window
#from src.clas_model import Classificacao
#from src.seg_model import Segmentacao
#from src.famacha import Famacha
import os

class Index(Widget):
    def stop_app(self):
        App.get_running_app().stop()

class Analise(BoxLayout):
    def open_file_chooser(self,title,dirselect):
        self.popup_title = title
        self.initial_path = os.path.join(os.path.expanduser('~'),'Pictures')
        file_chooser_popup = FileChooserPopup(callback=self.file_selected,title=self.popup_title,initial_path=self.initial_path,dirselect=dirselect)
        file_chooser_popup.open()

    def file_selected(self, file_path):
        """
        retorna -1,0,1
        """
        print("Arquivo selecionado na função principal:", file_path)
        #Classificação e resutados virao daqui

class FileChooserPopup(Popup):
    def __init__(self, callback,title,dirselect=False,initial_path = '', **kwargs):
        super(FileChooserPopup, self).__init__(**kwargs)
        self.title = title
        self.file_chooser = FileChooserListView(path=initial_path,dirselect=dirselect)
        self.file_chooser.bind(on_submit=self.file_selected)
        self.content = self.file_chooser
        self.callback = callback

    def file_selected(self, instance, selection, touch=None):
        print("Arquivo selecionado:", selection[0])
        self.callback(selection[0])  # Chama a função de retorno com o caminho do arquivo
        self.dismiss()  # Fecha o popup

class FamachApp(App):
    def build(self):
        self.title = "FAMACHAPP"
        self.icon = "../Imagens/logo.png"
        Window.maximize()
        #self.classific = Classificacao('src\\model_classific\\RF_Model.pkl')
        #self.segment = Segmentacao('src\\model_segment\\weights\\best.pt')
        #Famacha('src\\model_classific\\RF_Model.pkl','src\\model_segment\\weights\\best.pt')
        return Index()

if __name__ == "__main__":
    FamachApp().run()