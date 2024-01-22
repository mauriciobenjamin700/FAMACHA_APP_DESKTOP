from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.scrollview import ScrollView
from kivy.animation import Animation
#from src.clas_model import Classificacao
#from src.seg_model import Segmentacao
from src.famacha import Famacha

import os
import threading

process_image = ''
f = Famacha("Files/src/model_classific/RF_Model.pkl","Files/src/model_segment/weights/best.pt")


def inicia_analise(file_path):
    doente = -1
    resultado = f.segmentacao.segment_img(file_path)
    if type(resultado) != type(None):
        doente = 0
        if f.classificacao.predict(resultado):
            doente = 1
    return doente

class Index(Widget):    
    def stop_app(self):
        App.get_running_app().stop()
    
    def switch2confirmar_analise(self):
        self.ids.manager.current = "Confirmar_Analise"
    
    def switch2cartao_famacha(self):
        self.ids.manager.current = "Cartao_Famacha"
    
    def switch2analise(self):
        self.ids.manager.current = "Analise"
    
    def switch2imagem_em_analise(self):
        self.ids.manager.current = "Imagem_Em_Analise"

# Telas secundarias
class Analise(Screen):
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

        extensao = file_path.split('.')[-1]
        print(extensao)
        if extensao == 'jpg':
            global process_image 
            process_image = file_path
            myapp.app_switch2confirmar_analise()
            
        
        #Classificação e resultados virao daqui

class Cartao_Famacha(Screen):
    pass

class Confirmar_Analise(Screen):
    pass

class Imagem_Em_Analise(Screen):
    def analisar(self):
        print("Analise Iniciada\n")
        #resultado = inicia_analise(process_image)
        #print(resultado)

    def start_rotation_animation(self):
        # Criação da animação para girar a imagem continuamente
        anim = Animation(angle=360, duration=2)  # 360 graus em 2 segundos
        anim += Animation(angle=0)  # Volta à posição inicial
        anim.repeat = True  # Repete a animação continuamente
        anim.start(self.ids.rotating_image)  # Inicia a animação no objeto de imagem

# Funções auxiliares 
class CustomFileChooser(FileChooserListView):
    def keyboard_on_key_down(self, window, keycode, text, modifiers):
        # Intercepta a tecla "Esc" para impedir o fechamento do FileChooser
        if keycode[1] == 'escape':
            return True  # Impede o processamento padrão da tecla "Esc"
        return super(CustomFileChooser, self).keyboard_on_key_down(window, keycode, text, modifiers)

class FileChooserPopup(Popup):
    def __init__(self, callback,title,dirselect=False,initial_path = '', **kwargs):
        super(FileChooserPopup, self).__init__(**kwargs)
        self.title = title
        self.file_chooser = CustomFileChooser(path=initial_path,dirselect=dirselect)
        self.file_chooser.bind(on_submit=self.file_selected)
        self.content = self.file_chooser
        self.callback = callback

    def file_selected(self, instance, selection, touch=None):
        print("Arquivo selecionado:", selection[0])
        self.callback(selection[0])  # Chama a função de retorno com o caminho do arquivo
        self.dismiss()  # Fecha o popup

#Função que chama a aplicação
class FamachApp(App):
    def build(self):
        self.title = "FAMACHAPP"
        self.icon = "../Imagens/logo.png"
        Window.maximize()
        #self.classific = Classificacao('src\\model_classific\\RF_Model.pkl')
        #self.segment = Segmentacao('src\\model_segment\\weights\\best.pt')
        #Famacha('src\\model_classific\\RF_Model.pkl','src\\model_segment\\weights\\best.pt')
        self.index_instance = Index()
        return self.index_instance
    
    def app_switch2confirmar_analise(self):
        self.index_instance.switch2confirmar_analise()



if __name__ == "__main__":
    myapp = FamachApp()
    myapp.run()