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

from os.path import join, expanduser

from src import *

process_type = 1
path = ''
save_mode = ""

yolo =  SegModel(seg_model)

rf = PKL_Model(pre_model)



def analyze_image(file_path:str)->int:
    """
    Recebe o caminho para uma e retorna a situação FAMACHA da imagem \n
    Possíveis Retornos:
        - -1 (Caso não seja uma imagem válida)
        - 0 (Caso o animal esteja saudável)
        - 1 (Caso o animal esteja precisando de vermifugação)
    """
    global yolo
    global rf
    image = Image(file_path)
    famacha = -1
    new_image = resize(image,(512,512))
    resultado = Segment(new_image,yolo)
    if resultado is not None:
        famacha = PKL_classify(rf,Images2DF([resultado]))
    return famacha

def analyze_folder(file_path,mode):
    dataset = Folder(file_path)
    
    results = 0
    
    if dataset is not None:
        images = dataset[0]
        labels = dataset[1]
        
        if len(images) > 0: #caso tenhamos imagens para trabalhar
            
            global yolo
            global rf
            
            images = SegmentedList(images,yolo)
            
            df = Images2DF(images)
            
            predicts = PKL_classify(rf,df)
            
            # Criar um dicionário de dados onde as chaves são os nomes das colunas e os valores são os dados
            #data = {name: value for name, value in zip(labels, predicts)}
            data = list(zip(labels, predicts))
            print("Linha 64 -> ", data)
            # Criar o DataFrame a partir do dicionário de dados
            ending = DataFrame(data=data, columns=["Imagem","Status"])  # Supondo que você queira apenas uma linha no DataFrame
            print("Linha 67 ->", ending)
            if(save_results(ending,mode)):
                results = 1
    return results

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

    def switch2diagnostico_ruim(self):
        self.ids.manager.current = "Diagnostico_Ruim"

    def switch2diagnostico_bom(self):
        self.ids.manager.current = "Diagnostico_Bom"
    
    def switch2diagnostico_falho(self):
        self.ids.manager.current = "Diagnostico_Falho"

    def switch2salvar(self):
        self.ids.manager.current = "Salvar"
    
    def switch2diagnostico_pasta_bom(self):
        self.ids.manager.current = "Diagnostico_Pasta_Bom"

# Telas secundarias
class Analise(Screen):
    def open_file_chooser(self,title,dirselect):
        global process_type
        process_type = 0
        self.popup_title = title
        self.initial_path = join(expanduser('~'),'Pictures')
        file_chooser_popup = FileChooserPopup(callback=self.file_selected,title=self.popup_title,initial_path=self.initial_path,dirselect=dirselect)
        file_chooser_popup.open()

    def open_folder_chooser(self,title,dirselect):
        global process_type
        process_type = 1
        self.popup_title = title
        self.initial_path = join(expanduser('~'),'Pictures')
        file_chooser_popup = FileChooserPopup(callback=self.file_selected,title=self.popup_title,initial_path=self.initial_path,dirselect=dirselect)
        file_chooser_popup.open()

    def file_selected(self, file_path):
        """
        retorna -1,0,1
        """
        global path
        global process_type
        if process_type == 1:
            path = file_path
            myapp.app_switch2confirmar_analise()
        else:
            extensao = file_path.split('.')[-1]
            valid = ["jpg","jpeg","png", "JPG","JPEG","PNG"]
            if extensao in valid:
                path = file_path
                myapp.app_switch2confirmar_analise()
            
        
        #Classificação e resultados virao daqui

class Cartao_Famacha(Screen):
    pass

class Confirmar_Analise(Screen):
    def analisar(self):
        global process_type
        if process_type == 1:
            myapp.index_instance.switch2salvar()
        else:
            resultado = analyze_image(path)[0]
            print(resultado)
            if resultado == 1:
                myapp.index_instance.switch2diagnostico_ruim()
            elif resultado == 0:
                myapp.index_instance.switch2diagnostico_bom()
            else:
                myapp.index_instance.switch2diagnostico_falho()
        
class Imagem_Em_Analise(Screen):
    def enter_in_screen(self):
        global save_mode
        if save_mode == 'excel':
            self.save_excel()
        elif save_mode == 'csv':
            self.save_csv()
        elif save_mode == 'json':
            self.save_json()

    def save_excel(self):
        resultado = analyze_folder(path,"excel")
        if resultado == 1:
            myapp.index_instance.switch2diagnostico_pasta_bom()
        elif resultado == 0:
            myapp.index_instance.switch2diagnostico_falho()
        else:
            print(f"Deu ruim! {resultado}")
    def save_csv(self):
        resultado = analyze_folder(path,"csv")
        if resultado == 1:
            myapp.index_instance.switch2diagnostico_pasta_bom()
        elif resultado == 0:
            myapp.index_instance.switch2diagnostico_falho()
        else:
            print(f"Deu ruim! {resultado}")
    def save_json(self):
        resultado = analyze_folder(path,"json")
        if resultado == 1:
            myapp.index_instance.switch2diagnostico_pasta_bom()
        elif resultado == 0:
            myapp.index_instance.switch2diagnostico_falho()
        else:
            print(f"Deu ruim! {resultado}")

class Diagnostico_Ruim(Screen):
    pass

class Diagnostico_Bom(Screen):
    pass

class Diagnostico_Falho(Screen):
    pass

class Diagnostico_Pasta_Bom(Screen):
    pass

class Salvar(Screen):
    def save_excel(self):
        global save_mode
        save_mode = 'excel'
        myapp.index_instance.switch2imagem_em_analise()
    def save_csv(self):
        global save_mode
        save_mode = 'csv'
        myapp.index_instance.switch2imagem_em_analise()
    def save_json(self):
        global save_mode
        save_mode = 'json'
        myapp.index_instance.switch2imagem_em_analise()

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
        self.index_instance = Index()
        return self.index_instance
    
    def app_switch2confirmar_analise(self):
        self.index_instance.switch2confirmar_analise()



if __name__ == "__main__":
    myapp = FamachApp()
    myapp.run()